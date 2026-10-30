import re
from pathlib import Path

import numpy as np
import pyvista as pv

from yaml2nuctrans.meshing.mesh_1d import CreateMesh1D

M3_TO_L = 1000.0  # multiply a volume in m**3 by this to get liters

AXIS_NAMES = {0: "x", 1: "y", 2: "z"}  # OGS expression variable per axis


def porosity_at_source(units, source, medium_properties):
    """Auto-detect a repository source zone's porosity from the
    geological unit it physically lies in.

    Args:
        units (Units1D): Geological unit interface elevations.
        source (PointSource1D | LineSource1D): Source geometry.
        medium_properties (MediumProperties): Resolved OGS media, keyed
            by geological unit name.

    Returns:
        float: Porosity of the unit containing the source.

    Raises:
        ValueError: If the source zone does not fall entirely within one
            geological unit.
    """
    source_start, source_end = CreateMesh1D.get_source_location(source)
    vmin, vmax = sorted((source_start, source_end))

    sorted_interfaces = sorted(units.interfaces.items(), key=lambda kv: kv[1])
    names = [name for name, _ in sorted_interfaces]
    depths = [depth for _, depth in sorted_interfaces]
    unit_bounds = {names[i]: (depths[i], depths[i + 1]) for i in range(len(depths) - 1)}

    unit_key = CreateMesh1D.find_unit(vmin=vmin, vmax=vmax, unit_bounds=unit_bounds)
    if unit_key is None:
        raise ValueError(
            f"Source zone [{source_start}, {source_end}] does not fall "
            "entirely within a single geological unit; cannot "
            "auto-detect its porosity. Set "
            "initial_condition_nuclide.source_concentration.porosity "
            "explicitly instead."
        )

    rock_name = re.sub(r"_bottom$", "", unit_key)
    return medium_properties.get(rock_name).porosity.value


def compute_source_concentration(
    mesh,
    source,
    inventory_mol,
    porosity,
    area_repository,
    axis_index=2,
):
    """Compute the FEM nodal initial concentration for a repository source.

    The initial condition assigns a single concentration value to every
    mesh node located inside the source zone.

    Args:
        mesh (str | pathlib.Path | pyvista.DataSet): Path to the mesh used for the
            simulation.
        source (PointSource1D | LineSource1D): Source geometry.
        inventory_mol (float): Total nuclide inventory in the source zone,
            in moles.
        porosity (float): Porosity of the source zone, used to convert
            the inventory into a pore-water concentration.
        area_repository (float): Cross-sectional area represented by the
            1-D model, in m**2.
        axis_index (int, optional): Coordinate axis the 1-D model runs
            along. Defaults to 2 (z).

    Returns:
        float: Initial concentration in mol/L, for use as the FEM nodal initial condition.
    """
    if inventory_mol == 0.0:
        return 0.0

    source_start, source_end = CreateMesh1D.get_source_location(source)
    source_length = source_end - source_start

    if isinstance(mesh, (str, Path)):
        mesh = pv.read(str(mesh))

    coords = np.unique(np.round(mesh.points[:, axis_index], 9))
    below = coords[coords < source_start]
    above = coords[coords > source_end]

    if below.size == 0 or above.size == 0:
        raise ValueError(
            "Could not find a mesh node on both sides of the source "
            f"zone [{source_start}, {source_end}] along axis {axis_index}. "
            "The source must not reach the model boundary."
        )

    dist_below = source_start - below.max()
    dist_above = above.min() - source_end

    # an effective length that adds half of each neighboring element's length to the source zone's own length
    effective_length = source_length + 0.5 * dist_below + 0.5 * dist_above

    return inventory_mol / (porosity * area_repository * effective_length * M3_TO_L)


def build_source_concentration_ic(
    mesh,
    source,
    inventory_mol,
    porosity,
    area_repository,
    axis_index=2,
):
    """Build an OGS initial-condition (type, value) pair for a source.

    Args:
        mesh, source, inventory_mol, porosity, area_repository, axis_index:
            Forwarded to `compute_source_concentration`.

    Returns:
        dict: ``{"type": "Constant", "value": 0.0}`` if ``inventory_mol``
        is 0.0 (nothing to localize); otherwise ``{"type": "Function",
        "value": <expression string>}``.
    """
    if inventory_mol == 0.0:
        return {"type": "Constant", "value": 0.0}

    concentration = compute_source_concentration(
        mesh=mesh,
        source=source,
        inventory_mol=inventory_mol,
        porosity=porosity,
        area_repository=area_repository,
        axis_index=axis_index,
    )

    source_start, source_end = CreateMesh1D.get_source_location(source)
    axis_name = AXIS_NAMES[axis_index]

    expression = (
        f"if (({axis_name} >= {source_start}) and ({axis_name} <= {source_end})) "
        f"{concentration}; else 0.0;"
    )

    return {"type": "Function", "value": expression}
