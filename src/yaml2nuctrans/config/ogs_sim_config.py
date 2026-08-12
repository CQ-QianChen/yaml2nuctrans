from dataclasses import dataclass
from pathlib import Path

import yaml


Coordinate1D = float
Coordinate2D = tuple[float, float]
Coordinate3D = tuple[float, float, float]

@dataclass
class PointSource1D:
    """Store source geometry and location information.

    Attributes:
        loc (float): Source coordinate along the one-dimensional axis.
    """

    source_type: str
    loc: Coordinate1D

@dataclass
class LineSource1D:
    """Store source geometry and location information.

    Attributes:
        source_type (str): Source geometry type, for example "point".
        start_loc (float): Start coordinate of the source interval.
        end_loc (float): End coordinate of the source interval.
    """

    source_type: str
    start_loc: Coordinate1D
    end_loc: Coordinate1D

@dataclass
class PointSource2D:
    loc: Coordinate2D


@dataclass
class LineSource2D:
    start_loc: Coordinate2D
    end_loc: Coordinate2D


@dataclass
class SurfaceSource2D:
    """A 2D source region represented by polygon vertices."""
    vertices: list[Coordinate2D]


@dataclass
class PointSource3D:
    loc: Coordinate3D


@dataclass
class LineSource3D:
    start: Coordinate3D
    end: Coordinate3D


@dataclass
class SurfaceSource3D:
    """A 3D surface represented by three or more vertices."""
    vertices: list[Coordinate3D]


@dataclass
class VolumeSource3D:
    """A 3D volume, initially represented as an axis-aligned box."""
    min_corner: Coordinate3D
    max_corner: Coordinate3D
    

@dataclass
class Meshes:
    dimension: int
    domain: str
    top: str
    bottom: str
    reactive_domain: str

@dataclass
class NuclideInitialCondition:
    """Store initial-condition definitions for all nuclides.

    Attributes:
        types (list[str]): Initial-condition type for every nuclide.
        values (list[float]): Initial-condition value for every nuclide.
    """

    types: list[str]
    values: list[float]


@dataclass
class NuclideBoundaryCondition:
    """Store boundary-condition definitions for all nuclides.

    Attributes:
        types (list[str]): Boundary-condition type for every nuclide.
        meshes (list[str]): Mesh name for every nuclide boundary
            condition.
        values (list[str]): Boundary-condition value, parameter, or
            Python-function name for every nuclide.
    """

    types: list[str]
    meshes: list[str]
    values: list[str]

    def __post_init__(self):
        """Validate that every nuclide has one complete definition.

        Raises:
            ValueError: If condition lists do not have equal lengths.
        """
        lengths = {
            len(self.types),
            len(self.meshes),
            len(self.values),
        }

        if len(lengths) != 1:
            raise ValueError(
                "Nuclide boundary-condition lists must have the same "
                "number of entries."
            )


@dataclass
class PressureSetup:
    """Store initial and boundary pressures.

    Attributes:
        pressure_initial (float): Initial pressure in Pa.
        pressure_top (float): Pressure at the top boundary in Pa.
        pressure_bottom (float): Pressure at the bottom boundary in Pa.
    """

    pressure_initial: float
    pressure_top: float
    pressure_bottom: float


@dataclass
class ProcessCoupling:
    """Store enabled transport and reaction mechanisms.

    Attributes:
        decay (bool): Whether radioactive decay is enabled.
        diffusion (bool): Whether diffusion is enabled.
        advection (bool): Whether advection is enabled.
        sorption (bool): Whether sorption is enabled.
    """

    decay: bool
    diffusion: bool
    advection: bool
    sorption: bool


@dataclass
class ProcessSettings:
    """Store OGS process and nonlinear-solution settings.

    Attributes:
        nonlinear_solver_name (str): Name of the nonlinear solver.
        convergence_type (str): OGS nonlinear convergence criterion.
        norm_type (str): Norm used for convergence checks.
        relative_tolerances (list[float]): Relative tolerances, one per
            process variable.
        time_discretization (str): Time-discretization scheme.
    """

    nonlinear_solver_name: str
    convergence_type: str
    norm_type: str
    relative_tolerances: list[float]
    time_discretization: str


@dataclass
class FixedTimeStepping:
    """Store fixed time-stepping configuration.

    Attributes:
        time_stepping_type (str): OGS time-stepping type.
        t_initial (float): Initial simulation time in seconds.
        t_end (float): End simulation time in seconds.
        time_steps (list[float]): Time-step sizes in seconds.
        repeats (list[int]): Number of repeats for each time-step size.
    """

    time_stepping_type: str
    t_initial: float
    t_end: float
    time_steps: list[float]
    repeats: list[int]

@dataclass
class IterationNumberBasedTimeStepping:
    """Store iteration-number-based time-stepping configuration.

    Attributes:
        time_stepping_type (str): OGS time-stepping type.
        t_initial (float): Initial simulation time in seconds.
        t_end (float): End simulation time in seconds.
        initial_dt (int): Initial time-step size in seconds.
        minimum_dt (int): Minimum time-step size in seconds.
        maximum_dt (int): Maximum time-step size in seconds.
        number_iterations (list[int]): Number of iterations for each time-step size.
        multiplier (list[float]): Multiplier for each time-step size.
    """

    time_stepping_type: str
    t_initial: float
    t_end: float
    initial_dt: int
    minimum_dt: int
    maximum_dt: int
    number_iterations: list[int]
    multiplier: list[float]

@dataclass
class OutputSettings:
    """Store OGS output scheduling configuration.

    Attributes:
        repeats (list[int]): Number of output schedule repetitions.
        each_steps (list[int]): Output interval in time steps.
    """

    repeats: list[int]
    each_steps: list[int]


@dataclass
class NonLinearSolver:
    """Store nonlinear solver configuration.

    Attributes:
        name (str): Solver name used by process settings.
        solver_type (str): Nonlinear solver type, for example "Picard".
        max_iter (int): Maximum number of nonlinear iterations.
        linear_solver (str): Name of the associated linear solver.
    """

    name: str
    solver_type: str
    max_iter: int
    linear_solver: str


@dataclass
class LinearSolver:
    """Store one linear solver configuration.

    Attributes:
        name (str): Linear solver name.
        kind (str): Linear solver backend, for example "lis".
        prefix (str or None): Backend-specific options prefix.
        solver_type (str): Algorithm, for example "cg" or "SparseLU".
        precon_type (str): Preconditioner type.
        max_iteration_step (int): Maximum linear iterations.
        error_tolerance (float): Linear-solver tolerance.
    """

    name: list[str]
    kind: list[str]
    prefix: list[str]
    solver_type: list[str]
    precon_type: list[str]
    max_iteration_step: list[str]
    error_tolerance: list[float]


@dataclass
class TimeLoop:
    """Store all time-loop, process, solver, and output settings.

    Attributes:
        processes (ProcessSettings): OGS process settings.
        time_stepping (FixedTimeStepping): Fixed time-stepping setup.
        output (OutputSettings): Output scheduling settings.
    """

    processes: ProcessSettings
    time_stepping: FixedTimeStepping
    output: OutputSettings

@dataclass
class SimSetUp:
    """Store all time-loop, process, solver, and output settings.

    Attributes:
        timeloop (TimeLoop): Time-loop configuration.
        non_linear_solver (NonLinearSolver): Nonlinear solver setup.
        linear_solvers (LinearSolver): Available linear solvers.
    """

    timeloop: TimeLoop
    non_linear_solver: NonLinearSolver
    linear_solvers: LinearSolver

@dataclass
class SimulationConfig:
    """Store the complete OGS Python-boundary-condition configuration.

    Attributes:
        project_name (str): Use it for the OGS project name and output folder.
        create_python_bc (bool): Whether Python boundary conditions are
            enabled.
        python_bc_source_value (float): Constant concentration at the source location.
        create_meshes (bool): Whether meshes are created automatically.
        mesh_dimension (int): Mesh dimension, for example 1 for 1D.
        source (Source): Source geometry and location.
        process_variables (list[str]): Primary OGS process variables.
        secondary_variables (str): Secondary variables to output.
        initial_condition_nuclide (NuclideInitialCondition): Nuclide
            initial conditions.
        boundary_condition_nuclide (NuclideBoundaryCondition): Nuclide
            boundary conditions.
        pressure_setup (PressureSetup): Pressure initial and boundary
            conditions.
        process_options (ProcessCoupling): Enabled mechanisms.
        sim_setup (SimSetUp): Simulation setup configuration.
    """
    project_name: str
    create_python_bc: bool
    python_bc_source_value: float
    create_meshes: bool
    mesh_dimension: int
    meshes: Meshes
    quantity_of_interests: list
    source: PointSource1D | LineSource1D
    process_variables: list[str]
    secondary_variables: list[str]
    initial_condition_nuclide: NuclideInitialCondition
    boundary_condition_nuclide: NuclideBoundaryCondition
    pressure_setup: PressureSetup
    process_options: ProcessCoupling
    sim_setup: SimSetUp
    
    @classmethod
    def from_yaml(cls, filename):
        """Create simulation configuration from a YAML file.

        Args:
            filename (str or pathlib.Path): Configuration YAML path.

        Returns:
            SimulationConfig: Parsed simulation configuration.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If a required YAML key is missing.
        """
        filename = Path(filename)

        if not filename.is_file():
            raise FileNotFoundError(
                f"Configuration YAML file was not found: {filename}"
            )

        with filename.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)


        source_data = data["source"]
        mesh_dimension_from_data = int(data["mesh_dimension"])

        if source_data["type"] == "point" and mesh_dimension_from_data == 1:
            source_from_data = PointSource1D(source_type="point", loc=float(source_data["loc"]))
        if source_data["type"] == "line" and mesh_dimension_from_data == 1:
            source_from_data = LineSource1D(source_type="line", start_loc=float(source_data["start_loc"]),end_loc=float(source_data["end_loc"]))

        if ("mesh_names" not in data) and mesh_dimension_from_data == 1:
            meshes_from_data = Meshes(dimension=3, domain="domain",top="point_top", bottom="point_bottom", reactive_domain="reactive_domain")
        if ("mesh_names" in data) and mesh_dimension_from_data == 1:
            mesh_data = data["mesh_names"]
            meshes_from_data = Meshes(dimension=3, domain=mesh_data["domain"],top=mesh_data["top"], bottom=mesh_data["bottom["], reactive_domain=mesh_data["reactive_domain"])

        
        initial_data = data["initial_condition_nuclide"]
        boundary_data = data["boundary_condition_nuclide"]
        pressure_data = data["pressure_setup"]
        timeloop_data = data["sim_setup"]["timeloop"]
        process_data = timeloop_data["processes"]

        number_of_nuclide = len(initial_data["value"])
        if ("relative_tol" not in process_data):
            process_data_relative_tol = [1e-16]*(number_of_nuclide+1)
        else:
            process_data_relative_tol = process_data_relative_tol

       
        time_stepping_data = process_data["time_stepping"]

        if time_stepping_data["type"] == "FixedTimeStepping":
            time_stepping_data_resolved = FixedTimeStepping(
                    time_stepping_type=time_stepping_data["type"],
                    t_initial=float(
                        time_stepping_data["t_initial"]
                    ),
                    t_end=float(time_stepping_data["t_end"]),
                    time_steps=[
                        float(value)
                        for value in time_stepping_data["time_step"]
                    ],
                    repeats=[
                        int(value)
                        for value in time_stepping_data["repeat"]
                    ],
                )
        if time_stepping_data["type"] == "IterationNumberBasedTimeStepping":
            time_stepping_data_resolved = IterationNumberBasedTimeStepping(
                    time_stepping_type=time_stepping_data["type"],
                    t_initial=float(
                        time_stepping_data["t_initial"]
                    ),
                    t_end=float(time_stepping_data["t_end"]),
                    initial_dt=int(
                        time_stepping_data["initial_dt"]
                    ),
                    minimum_dt=int(
                        time_stepping_data["minimum_dt"]
                    ),
                    maximum_dt=int(
                        time_stepping_data["maximum_dt"]
                    ),
                    number_iterations=[
                        int(value)
                        for value in time_stepping_data["number_iterations"]
                    ],
                    multiplier=[
                        float(value)
                        for value in time_stepping_data["multiplier"]
                    ],
                )


        output_data = timeloop_data["output"]
        nonlinear_data = data["sim_setup"]["non_linear_solver"]
        linear_data = data["sim_setup"]["linear_solver"]

        linear_solvers = LinearSolver(
                            name=linear_data["name"],
                            kind=linear_data["kind"],
                            prefix=linear_data["prefix"],
                            solver_type=linear_data["solver_type"],
                            precon_type=linear_data["precon_type"],
                            max_iteration_step=linear_data["max_iteration_step"],
                            error_tolerance=linear_data["error_tolerance"]
                            )

        return cls(
            project_name=str(data["project_name"]),
            create_python_bc=bool(data["create_python_bc"]),
            python_bc_source_value=float(data["python_bc_source_value"]),
            create_meshes=bool(data["create_meshes"]),
            mesh_dimension = mesh_dimension_from_data,
            meshes = meshes_from_data,
            quantity_of_interests = data["quantity_of_interests"],
            source=source_from_data,
            process_variables=list(data["process_variables"]),
            secondary_variables=data["secondary_variables"],
            initial_condition_nuclide=NuclideInitialCondition(
                types=list(initial_data["type"]),
                values=[
                    float(value)
                    for value in initial_data["value"]
                ],
            ),
            boundary_condition_nuclide=NuclideBoundaryCondition(
                types=list(boundary_data["type"]),
                meshes=list(boundary_data["mesh"]),
                values=list(boundary_data["value"]),
            ),
            pressure_setup=PressureSetup(
                pressure_initial=float(
                    pressure_data["pressure_initial"]
                ),
                pressure_top=float(
                    pressure_data["pressure_top"]
                ),
                pressure_bottom=float(
                    pressure_data["pressure_bottom"]
                ),
            ),
            process_options=ProcessCoupling(
                decay=bool(data["decay"]),
                diffusion=bool(data["diffusion"]),
                advection=bool(data["advection"]),
                sorption=bool(data["sorption"]),
            ),
            sim_setup=SimSetUp(
                timeloop=TimeLoop(
                    processes=ProcessSettings(
                        nonlinear_solver_name=process_data[
                            "nonlinear_solver_name"
                        ],
                    convergence_type=process_data[
                        "convergence_type"
                    ],
                    norm_type=process_data["norm_type"],
                    relative_tolerances=process_data_relative_tol,
                    time_discretization=process_data[
                        "time_discretization"
                    ],
                ),
                time_stepping=time_stepping_data_resolved,
                output=OutputSettings(
                    repeats=[
                        int(value)
                        for value in output_data["repeat"]
                    ],
                    each_steps=[
                        int(value)
                        for value in output_data["each_steps"]
                    ],
                )),
                non_linear_solver=NonLinearSolver(
                    name=nonlinear_data["name"],
                    solver_type=nonlinear_data["type"],
                    max_iter=int(nonlinear_data["max_iter"]),
                    linear_solver=nonlinear_data["linear_solver"],
                ),
                linear_solvers=linear_solvers)
        )
