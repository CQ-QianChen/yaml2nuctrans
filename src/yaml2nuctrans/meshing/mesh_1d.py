import gmsh
import numpy as np
import matplotlib.pyplot as plt
import os
import ogstools as ot
from pathlib import Path
import subprocess
from yaml2nuctrans.config.ogs_sim_config import PointSource1D, LineSource1D, PointSource2D, LineSource2D, SurfaceSource2D, PointSource3D, LineSource3D, SurfaceSource3D, VolumeSource3D
from yaml2nuctrans.primary_parameters.geometry import Units1D

class CreateMesh1D:
    """Class to create a 1D mesh for simulation."""

    def __init__(self):
        pass

    @staticmethod
    def get_source_location(source):
        """Function to get location of the source.

        Args:
            source (PointSource1D | LineSource1D): The source definition.

        Returns:
            source_start, source_end: starting and ending locations for the source.
        """
        if isinstance(source, PointSource1D):
            return source.loc, source.loc

        if isinstance(source, LineSource1D):
            return source.start_loc, source.end_loc

        raise TypeError(
            "source must be PointSource1D or LineSource1D, "
            f"not {type(source).__name__}."
        )


    @staticmethod
    def generate_1d_grid(units, source, dz_min=1.0, dz_max=20.0, growth_rate=1.08):
        """Generate a 1D graded grid with fine spacing near source.

        Args:
            units (Units1D): {unit_top: location, unit_name1: location, unit_name2: location}
            source (PointSource1D | LineSource1D): The source definition.
            dz_min (float, optional): Smallest point step size (near the source). Defaults to 1.0.
            dz_max (float, optional): Largest point step size (far away). Defaults to 20.0.
            growth_rate (float, optional): Controls how fast the point size grows away from the source. Defaults to 1.08.

        Returns:
            points (np.array): 1D graded grid with fine spacing near source.
        """

        # start from the source
        source_start, source_end = CreateMesh1D.get_source_location(source)
        units_interfaces = units.interfaces

        unit_depths = units_interfaces.values()
        unit_top = max(unit_depths)
        unit_bottom = min(unit_depths)
        # Upstream from source
        dz = dz_min
        points_up = [source_start]
        while points_up[-1] < unit_top:
            # dz_n+1​=min(dz_n​⋅r,dz_max​)
            dz = min(dz * growth_rate, dz_max)
            points_new = points_up[-1] + dz
            if points_new > unit_top:
                points_new = unit_top
            points_up.append(points_new)

        # Downstream from source
        dz = dz_min
        points_down = [source_end]
        while points_down[-1] > unit_bottom:
            # dz_n+1​=min(dz_n​⋅r,dz_max​)
            dz = min(dz * growth_rate, dz_max)
            points_new = points_down[-1] - dz
            if points_new < unit_bottom:
                points_new = unit_bottom
            points_down.append(points_new)

        points = np.unique(np.concatenate([points_down, points_up, list(unit_depths)]))

        return np.sort(points)

    @staticmethod
    def find_unit(vmin, vmax, unit_bounds):
        """Find which unit the range [vmin, vmax] falls into.

        Args:
            vmin (float): Lower bound of the range.
            vmax (float): Upper bound of the range.
            unit_bounds (dict): Unit intervals {unit_name1: (x0, x1), unit_name2: (x2, x3)}.

        Returns:
            unit_name (str) or None: The unit which contains the given range.
        """
        for unit_name, (low, high) in unit_bounds.items():
            if low <= vmin and vmax <= high:
                return unit_name
        return None

    @staticmethod
    def piecewise_linear_meshsize(
        loc, control_start, control_end, source_start, source_end, mz_min=0.005, mz_max=5
    ):
        """Compute local piecewise-linear mesh size per point node.

        Args:
            loc (float): Location of the point node.
            control_start (float): Refinement transitions point near the start location of the source. If loc >=control_start, mesh size set to mz_max.
            control_end (float): Refinement transitions point near the end location of the source. If loc <=control_end, mesh size set to mz_max.
            source_start (float): start location of the source.
            source_end (float): end location of the source.
            mz_min (float, optional): Smallest mesh size. Defaults to 0.005.
            mz_max (float, optional): Largest mesh size. Defaults to 5.

        Returns:
            float: mesh size
        """
        if loc <= control_end:
            return mz_max
        elif loc <= source_end:
            return (mz_min - mz_max) * loc / (source_end - control_end) + (
                mz_max * source_end - mz_min * control_end
            ) / (source_end - control_end)
        elif loc <= source_start:
            return mz_min
        elif loc <= control_start:
            return (mz_max - mz_min) * loc / (control_start - source_start) + (
                mz_min * control_start - mz_max * source_start
            ) / (control_start - source_start)
        else:
            return mz_max

    @staticmethod
    def generate_1d_meshes(
        units,
        source,
        output_msh_path,
        dz_min=1,
        dz_max=200,
        growth_rate=1.5,
        mz_min=0.01,
        mz_max=1,
        control_start2source=10,
        control_end2source=10,
    ):
        """Create a 1D meshes to be used in a OGS simulation.

        Args:
            units (Units1D): {unit_top: location, unit_name1: location, unit_name2: location}
            source (PointSource1D | LineSource1D): The source definition.
            output_msh_path (_type_): _description_
            dz_min (float, optional): Smallest point step size (near the source). Defaults to 1.
            dz_max (float, optional): Largest point step size (far away). Defaults to 200.
            growth_rate (float, optional): Controls how fast the point size grows away from the source. Defaults to 1.5.
            mz_min (float, optional): Smallest mesh size. Defaults to 0.01.
            mz_max (float, optional): Largest mesh size. Defaults to 1.
            control_start2source (float, optional): Distance to the start location of the source (for calculating the refinement transition point). Defaults to 10.
            control_end2source (float, optional): Distance to the end location of the source (for calculating the refinement transition point). Defaults to 10.
        """

        Path(output_msh_path).mkdir(parents=True,exist_ok=True)

        # prepare the unit intervals
        units_interfaces = units.interfaces
        units_sorted = sorted(units_interfaces.items(), key=lambda kv: kv[1])
        unit_keynames = [u[0] for u in units_sorted]
        unit_depths = [u[1] for u in units_sorted]
        unit_bounds = {
            unit_keynames[i]: (unit_depths[i], unit_depths[i + 1])
            for i in range(len(unit_depths) - 1)
        }
        unit_names = list(unit_bounds.keys())

        source_start, source_end = CreateMesh1D.get_source_location(source)
        # create node locations
        point_nodes = CreateMesh1D.generate_1d_grid(
            units=units,
            source=source,
            dz_min=dz_min,
            dz_max=dz_max,
            growth_rate=growth_rate,
        )

        # Gmsh setup
        gmsh.initialize()
        gmsh.model.add("mesh")

        # Add points based on the coordinates
        point_tags = []
        for point in point_nodes:
            mz = CreateMesh1D.piecewise_linear_meshsize(
                point,
                control_start=source_start + control_start2source,
                control_end=source_end - control_end2source,
                source_start=source_start,
                source_end=source_end,
                mz_min=mz_min,
                mz_max=mz_max,
            )
            point_tag = gmsh.model.geo.addPoint(0, 0, point, mz)
            point_tags.append(point_tag)

        # Create line segments connecting the points
        lines_in_units = {unit_name: [] for unit_name in unit_names}
        for i in range(len(point_tags) - 1):
            line_tag = gmsh.model.geo.addLine(point_tags[i], point_tags[i + 1])
            lines_in_units[
                CreateMesh1D.find_unit(
                    vmin=point_nodes[i], vmax=point_nodes[i + 1], unit_bounds=unit_bounds
                )
            ].append(line_tag)

        gmsh.model.geo.synchronize()

        # add point physical group
        point_top = gmsh.model.addPhysicalGroup(0, [point_tags[-1]])
        gmsh.model.setPhysicalName(0, point_top, "point_top")

        # add unit physical group
        for unit_name, lines_in_unit in lines_in_units.items():
            lines_phys = gmsh.model.addPhysicalGroup(1, lines_in_unit)
            gmsh.model.setPhysicalName(1, lines_phys, unit_name)

        point_bottom = gmsh.model.addPhysicalGroup(0, [point_tags[0]])
        gmsh.model.setPhysicalName(0, point_bottom, "point_bottom")

        # finalize and mesh
        gmsh.model.mesh.generate(1)
        gmsh.write(os.path.join(output_msh_path, "mesh_1d.msh"))
        gmsh.finalize()

        meshes = ot.Meshes.from_gmsh(
            filename=os.path.join(output_msh_path, "mesh_1d.msh"), dim=1, reindex=True, log=False
        )
        meshes.save(meshes_path=Path(output_msh_path), overwrite=True)
        original_dir = os.getcwd()
        os.chdir(output_msh_path)
        domain_vtu = "domain.vtu"
        output_prefix = "reactive_"

        subprocess.run(
            [
                "identifySubdomains",
                "-m", domain_vtu,
                "-s", "1e-6",
                "-o", output_prefix,
                "--",
                domain_vtu,
            ],
            check=True,
        )
        os.chdir(original_dir)

    @staticmethod
    def plot_units_and_source(units, source):
        """Plot 1D unit boundary locations and source location(s).

        Args:
            units (Units1D): {unit_top: location, unit_name1: location, unit_name2: location}
            source (PointSource1D | LineSource1D): The source definition.
        """
        plt.figure(figsize=(2, 10))
    
        units_sorted = sorted(units.interfaces.items(), key=lambda kv: kv[1])
        unit_z = [u[1] for u in units_sorted]

        # Plot unit boundaries
        plt.scatter([0] * len(unit_z), unit_z, marker="_", s=500)
        for name, loc in units_sorted:
            plt.text(0.15, loc, name, ha="center")

        # Plot source
        if source["type"] == "point":
            plt.scatter([0], [source["loc"]], marker="*", s=50, color="red")
            plt.text(-0.02, source["loc"], "Source", ha="center")

        elif source["type"] == "line":
            plt.plot(
                [0, 0], [source["start_loc"], source["end_loc"]], linewidth=2, color="red"
            )
            plt.text(
                -0.02,
                0.5 * (source["start_loc"] + source["end_loc"]),
                "Source",
                ha="center",
            )
        else:
            raise ValueError("Invalid source type. Must be 'point' or 'line'.")

        plt.xticks([])
        plt.ylabel("Depth")
        plt.title("Unit bounds and source location (1D)")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
