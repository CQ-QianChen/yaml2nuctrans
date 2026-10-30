from pathlib import Path

import numpy as np
import ogstools as ot
import pyvista as pv
import radioactivedecay as rd
from scipy.integrate import cumulative_trapezoid

M3_TO_L = 1000.0  # 1 m**3 = 1000 L
SECONDS_PER_YEAR = 31557600.0

# German EndlSiAnfV repository safety requirements on release from the
# containment-providing rock zone:
LIMIT_CUMULATIVE_1E6A = 1e-4  # proportion of initial inventory, at t = 1e6 a
LIMIT_RATE_PER_A = 1e-9  # proportion of initial inventory, per year

# Zone names auto-derived by `ewg_zones_from_host_rock`.
DOMAIN_ZONE = "domain"
HOST_ROCK_ZONE = "host rock"
ABOVE_HOST_ROCK_ZONE = "above host rock"
BELOW_HOST_ROCK_ZONE = "below host rock"

CONCENTRATION_FLOOR = 0.0
DISCHARGE_FLOOR = 0.0


def _zone_material_ids(zone_spec, medium_ids):
    """Load a zone specification to a list of int OGS MaterialIDs.

    Args:
        zone_spec (str | int | list[str | int]): One or more geological
            unit names and/or OGS medium IDs that make up the zone.
        medium_ids (MediumIds | dict[str, int], optional): Used to
            convert a geological unit name to its OGS medium ID.
            Only needed when `zone_spec` contains unit names.

    Returns:
        list[int]: MaterialIDs belonging to the zone.
    """
    if isinstance(zone_spec, (str, int)):
        zone_spec = [zone_spec]

    material_ids = []
    for entry in zone_spec:
        if isinstance(entry, str):
            layer_id = (
                medium_ids.id_for(entry)
                if hasattr(medium_ids, "id_for")
                else medium_ids[entry]
            )
            material_ids.append(int(layer_id))
        else:
            material_ids.append(int(entry))

    return material_ids


def ewg_zones_from_host_rock(domain_mesh, host_rock, medium_ids=None):
    """Auto-partition a domain mesh's MaterialIDs into the zones needed
    for an ewg discharge check, given only the host-rock unit(s).
    MaterialIDs are assumed to be assigned bottom-to-top in increasing
    order

    Args:
        domain_mesh (str | Path | pyvista.DataSet): Path to the domain
            mesh (carrying "MaterialIDs")
        host_rock (str | int | list[str | int]): One or more geological
            unit names and/or OGS medium IDs that make up the
            containment-providing rock zone (ewg).
        medium_ids (MediumIds | dict[str, int], optional): Needed only
            when `host_rock` references geological unit names rather
            than raw medium IDs.

    Returns:
        tuple[pyvista.DataSet, dict[str, list[int]]]: The loaded domain
        mesh, and `{"domain": [...], "host rock": [...], "above host rock":
        [...], "below host rock": [...]}`, each a list of MaterialIDs.
    """
    if isinstance(domain_mesh, (str, Path)):
        domain_mesh = pv.read(str(domain_mesh))

    material_ids = np.asarray(domain_mesh["MaterialIDs"])
    host_ids = set(_zone_material_ids(host_rock, medium_ids))

    all_ids = sorted({int(i) for i in np.unique(material_ids)})
    below_ids = [i for i in all_ids if i < min(host_ids)]
    above_ids = [i for i in all_ids if i > max(host_ids)]

    zones = {
        DOMAIN_ZONE: all_ids,
        HOST_ROCK_ZONE: sorted(host_ids),
        ABOVE_HOST_ROCK_ZONE: above_ids,
        BELOW_HOST_ROCK_ZONE: below_ids,
    }
    return domain_mesh, zones


def compute_zone_substance_amounts(
    result_pvd_path,
    domain_mesh,
    nuclide_names,
    porosity_by_material_id,
    area_repository,
    zones,
    medium_ids=None,
):
    """Compute the per-zone substance amount (mol) of every nuclide,
    over the whole simulated time series.

    Args:
        result_pvd_path (str | Path): Path to the OGS `.pvd` result file.
        domain_mesh (str | Path | pyvista.DataSet): Path to the static
            domain mesh (carrying "MaterialIDs").
        nuclide_names (list[str]): Point-data field name of every
            decay-chain member, in the same order as `NuclideData.nuclide_names`.
        porosity_by_material_id (dict[int, float]): Porosity of each
            geological layer.
        area_repository (float): Cross-sectional area represented by the
            1-D model, in m**2.
        zones (dict[str, str | int | list[str | int]]): Zone name to the
            OGS medium ID(s) and/or geological unit name(s) that make up
            that zone, e.g. `{"host rock": ["Claystone"],
            "overburden": [6, 7, 8, 9]}`.
        medium_ids (MediumIds | dict[str, int], optional): Needed only
            when `zones` references geological unit names rather than
            raw medium IDs.

    Returns:
        dict[str, dict[tuple, list[float]]]: `{f"{nuclide_name}
        _substance_amount": {(zone_name, t): [amount], ...}, ...}`.
    """
    if isinstance(domain_mesh, (str, Path)):
        # # loads it as a mesh object (a pyvista-based UnstructuredGrid-like object). independent of the simulation results
        domain_mesh = pv.read(str(domain_mesh))
    # Extracts the MaterialIDs array from the domain mesh's data
    material_ids = np.asarray(domain_mesh["MaterialIDs"])
    # computes geometric size measures for every cell and attaches them as cell-data arrays
    cell_sizes = domain_mesh.compute_cell_sizes()
    # "Length" for 1-D cells (line elements)
    measure = np.asarray(cell_sizes["Length"])
    # producing a per-cell porosity value by looking up each cell's material ID in the POROSITY dict.
    porosity = np.vectorize(porosity_by_material_id.get)(material_ids).astype(float)
    # Combines everything into a per-cell pore volume in liters
    weight = porosity * measure * area_repository * M3_TO_L

    zone_masks = {
        zone_name: np.isin(material_ids, _zone_material_ids(zone_spec, medium_ids))
        for zone_name, zone_spec in zones.items()
    }
    # Loads the full time-series simulation output
    mesh_series = ot.MeshSeries(str(result_pvd_path))
    # one per output timestep
    time = mesh_series.timevalues

    results = {f"{nuclide_name}_substance_amount": {} for nuclide_name in nuclide_names}

    for t, mesh in zip(time, mesh_series):

        for nuclide_name in nuclide_names:
            # Clips away small negative numerical under/overshoot in the raw
            # FE concentration field before it is averaged onto cells
            mesh.point_data[nuclide_name] = np.clip(
                mesh.point_data[nuclide_name], CONCENTRATION_FLOOR, None
            )
        # converts all point-based fields of that mesh into cell-based fields (by averaging each cell's nodal values)
        cell_data = mesh.point_data_to_cell_data()
        for nuclide_name in nuclide_names:
            concentration = cell_data[nuclide_name]
            for zone_name, mask in zone_masks.items():
                # c[mask] selects the cell-averaged concentrations in that zone, weight[mask] selects the corresponding cell pore-volumes (in liters)
                # c[mask] * weight[mask] concentration × weight gives moles in that cell
                results[f"{nuclide_name}_substance_amount"][(zone_name, float(t))] = [
                    float(np.sum(concentration[mask] * weight[mask]))
                ]

    return results


def _field_time_series(zone_label, times, values):
    """Build a `{(zone_label, t): [value], ...}` dict for one field."""
    return {(zone_label, t): [float(v)] for t, v in zip(times, values)}


def _select_fields(
    times, discharged, frac_discharged, release_rate, fields, zone_label
):
    """Build {field_name: {(zone_label, t): [value], ...}, ...}."""
    available = {
        "discharged": discharged,
        "frac_discharged": frac_discharged,
        "release_rate": release_rate,
    }
    return {
        field_name: _field_time_series(zone_label, times, available[field_name])
        for field_name in fields
    }


BASE_DISCHARGE_FIELDS = ("discharged", "frac_discharged", "release_rate")


def _split_fields(fields):
    """Split a flat `fields` list into the per-nuclide base field names."""
    per_nuclide = tuple(f for f in fields if f in BASE_DISCHARGE_FIELDS)
    total = tuple(
        f.removeprefix("total_")
        for f in fields
        if f.removeprefix("total_") in BASE_DISCHARGE_FIELDS and f.startswith("total_")
    )
    return per_nuclide, total


def compute_ewg_discharge(
    substance_amounts,
    domain_zone,
    host_rock_zone,
    fields=(
        "discharged",
        "frac_discharged",
        "release_rate",
        "total_discharged",
        "total_frac_discharged",
        "total_release_rate",
    ),
):
    """Compute the cumulative discharge amount, fraction, and release
    rate from the ewg, for every decay-chain nuclide and for their sum,
    via a mass balance on the host rock zone's own substance amount.

    Args:
        substance_amounts (dict): Output of
            `compute_zone_substance_amounts`: `{f"{nuclide_name}
            _substance_amount": {(zone_name, t): [amount], ...}, ...}`,
            including an entry for `host_rock_zone`.
        domain_zone (str): Zone name holding the amount originally
            present in the whole domain, used as the t=0 normalization
            reference.
        host_rock_zone (str): Zone name holding the containment-providing
            rock zone's (ewg's) own substance amount over time.
        fields (list[str] | tuple[str]): Which QoIs to save into the
            results.

    Returns:
        dict[str, dict[tuple, list[float]]]: {f"{nuclide_name}
        _{field_name}": {(host_rock_zone, t): [value], ...}, ...}, plus
        "total_{field_name}".
    """
    per_nuclide_fields, total_fields = _split_fields(fields)

    results = {}
    total_discharged = None
    total_n_ini_ref = 0.0
    times = None
    times_years = None

    for qoi_name, zone_series in substance_amounts.items():
        nuclide_name = qoi_name.removesuffix("_substance_amount")
        times = sorted({t for _, t in zone_series})
        times_years = np.array(times) / SECONDS_PER_YEAR

        n_ini_ref = zone_series[(domain_zone, times[0])][0]
        if n_ini_ref == 0.0:
            continue

        host_rock_amount = np.array(
            [zone_series[(host_rock_zone, t)][0] for t in times]
        )
        decay_constant_per_year = (
            np.log(2) / rd.Nuclide(nuclide_name).half_life() * SECONDS_PER_YEAR
        )
        # cumulative_decay_loss(t)
        decay_loss_rate = decay_constant_per_year * host_rock_amount
        cumulative_decay_loss = cumulative_trapezoid(
            decay_loss_rate, times_years, initial=0.0
        )

        # Integrates the mass-balance identity `dN/dt = -decay_constant * N(t) - release_rate_abs(t)`
        # over [0, t] (`discharged(t) = N(0) - N(t) - cumulative_decay_loss(t)`)
        discharged = np.clip(
            host_rock_amount[0] - host_rock_amount - cumulative_decay_loss,
            DISCHARGE_FLOOR,
            None,
        )
        frac_discharged = discharged / n_ini_ref
        release_rate = np.gradient(discharged, times_years) / n_ini_ref

        for field_name, series in _select_fields(
            times,
            discharged,
            frac_discharged,
            release_rate,
            per_nuclide_fields,
            host_rock_zone,
        ).items():
            results[f"{nuclide_name}_{field_name}"] = series

        total_discharged = (
            discharged if total_discharged is None else total_discharged + discharged
        )
        total_n_ini_ref += n_ini_ref

    if total_fields and total_n_ini_ref > 0.0:
        total_frac_discharged = total_discharged / total_n_ini_ref
        total_release_rate = (
            np.gradient(total_discharged, times_years) / total_n_ini_ref
        )

        for field_name, series in _select_fields(
            times,
            total_discharged,
            total_frac_discharged,
            total_release_rate,
            total_fields,
            host_rock_zone,
        ).items():
            results[f"total_{field_name}"] = series

    return results
