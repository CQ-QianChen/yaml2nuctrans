# python_boundary_condition.py
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from yaml2nuctrans.config.ogs_sim_config import SimulationConfig
from yaml2nuctrans.config.ogs_sim_config import PointSource1D


@dataclass
class PythonBoundaryConditionScript:
    """Create the default OGS Python concentration boundary condition.

    Attributes:
        output_path: Full output path of the generated Python file.
        variable_name: Variable name referenced by the OGS project,
            for example ``"bc_c"``.
        source_bc: Concentration imposed at the source location.
    """

    output_path: None

    @classmethod
    def from_config(
        cls,
        config,
        output_path,
    ):
        """Create a Python BC script definition from simulation config.

        Args:
            config (SimulationConfig): Parsed simulation configuration.
            output_path (str | Path): Full path for the generated Python script.

        Returns:
            PythonBoundaryConditionScript: Script definition ready to
            be written.

        Raises:
            ValueError: If Python boundary conditions are disabled,
                absent, or ambiguously configured.
            TypeError: If the source is not a 1D point source.
        """

        output_path = Path(output_path)
        if config.create_python_bc:
            
            if config.mesh_dimension != 1:
                raise ValueError(
                    "The default Python BC script currently supports "
                    "only a 1D mesh."
                )

            if not isinstance(config.source, PointSource1D):
                raise TypeError(
                    "The default Python BC script requires a "
                    "PointSource1D source."
                )

            boundary_condition = (
                config.boundary_condition_nuclide
            )

            python_indices = [
                index
                for index, condition_type in enumerate(
                    boundary_condition.types
                )
                if condition_type.lower() == "python"
            ]

            if not python_indices:
                raise ValueError(
                    "No nuclide boundary condition has type 'Python'."
                )

            python_index = python_indices[0]
            variable_name = boundary_condition.values[python_index]
            source_bc = config.python_bc_source_value
            source_loc = config.source.loc

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            script = dedent(f"""\
            try:
                import ogs.callbacks as OpenGeoSys
            except ModuleNotFoundError:
                import OpenGeoSys


            class BCConcentration(OpenGeoSys.BoundaryCondition):
                def getDirichletBCValue(
                    self,
                    _t,
                    coords,
                    _node_id,
                    _primary_vars,
                ):
                    _x, _y, z = coords

                    if z == 0.0:
                        return (True, 0.0)

                    if z == {source_loc}:
                        # Continuous source release.
                        return (True, {source_bc})

                    # No Dirichlet boundary condition.
                    return (False, 0.0)


            {variable_name} = BCConcentration()
            """)

            output_path.write_text(
                script,
                encoding="utf-8",
            )

            output_path = Path(output_path).name # relative to the project file

        return cls(output_path=output_path)