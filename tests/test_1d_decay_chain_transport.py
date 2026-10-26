import shutil

import numpy as np
import pytest
import matplotlib.pyplot as plt

from yaml2nuctrans.config.generate_ogs_prj import generate_ogsprj
from yaml2nuctrans.config.ogs_sim_config import (
    FixedTimeStepping,
    LinearSolver,
    Meshes,
    NonLinearSolver,
    NuclideBoundaryCondition,
    NuclideInitialCondition,
    OutputSettings,
    PointSource1D,
    PressureSetup,
    ProcessCoupling,
    ProcessSettings,
    SimSetUp,
    SimulationConfig,
    TimeLoop,
)
from yaml2nuctrans.get_model.analytical_solution import Analytical_Sol
from yaml2nuctrans.ogs_parameters.nuclide_properties import (
    DiffusionModel,
    default_pore_diffusion,
)
from yaml2nuctrans.read_results.read_ogs_model_results import read_vtu_field

pytestmark = pytest.mark.skipif(
    shutil.which("ogs") is None,
    reason="OGS executable not available on PATH",
)

SECONDS_PER_YEAR = 31557600.0  # Julian year
COMPARISON_TIME_YEARS = 1_000_000.0

WATER_DYNAMIC_VISCOSITY = 1.0e-3  # Pa*s
WATER_DENSITY = 1000.0  # kg/m^3
GRAVITATIONAL_ACCELERATION = 9.80665  # m/s^2
TARGET_DARCY_FLUX = 1.0e-3 / SECONDS_PER_YEAR  # 1 mm/yr in m/s

CONCENTRATION_FLOOR = 1e-80


def _pressure_gradient_for_target_flux(permeability):
    """Solve Darcy's law, q = -(kappa/mu)*(dP/dx - rho_f*g), for dP/dx.

    x is the distance from the source, directed into the domain, so
    dP/dx and g are both taken as scalar components along that
    direction.
    """
    gravity_term = WATER_DENSITY * GRAVITATIONAL_ACCELERATION
    return gravity_term - TARGET_DARCY_FLUX * WATER_DYNAMIC_VISCOSITY / permeability


def _build_sim_config(
    rock_data,
    rock_name,
    units_1d,
    source,
    nuclide_names,
    c_init_value,
    c_inlet_value,
    project_name,
):
    """A SimulationConfig for the fully coupled Claystone decay-chain
    problem: point source at 0 m, Dirichlet concentration = c_inlet_value
    at the source, initial concentration = c_init_value everywhere, and
    a darcy flux of 1 mm/yr.
    """
    rock_unit = rock_data.get(rock_name)
    permeability = rock_unit.get_property("intrinsic_permeability").value_for_run
    pressure_gradient = _pressure_gradient_for_target_flux(permeability)

    domain_thickness = max(units_1d.interfaces.values()) - min(
        units_1d.interfaces.values()
    )
    pressure_top = 100_000.0
    pressure_bottom = pressure_top + pressure_gradient * domain_thickness

    number_of_nuclides = len(nuclide_names)

    return SimulationConfig(
        project_name=project_name,
        create_python_bc=False,
        python_bc_source_value=0.0,
        create_meshes=True,
        mesh_dimension=1,
        meshes=Meshes(
            dimension=3,
            domain="domain",
            top="point_top",
            bottom="point_bottom",
            reactive_domain="reactive_domain",
        ),
        quantity_of_interests=["concentration"],
        source=PointSource1D(source_type="point", loc=source.loc),
        process_variables=["pressure", "concentration"],
        secondary_variables=["Flux"],
        initial_condition_nuclide=NuclideInitialCondition(
            types=["Constant"] * number_of_nuclides,
            values=[c_init_value] * number_of_nuclides,
        ),
        boundary_condition_nuclide=NuclideBoundaryCondition(
            types=["Dirichlet"] * number_of_nuclides,
            meshes=["point_top"] * number_of_nuclides,
            values=[c_inlet_value] * number_of_nuclides,
        ),
        pressure_setup=PressureSetup(
            pressure_initial=pressure_top,
            pressure_top=pressure_top,
            pressure_bottom=pressure_bottom,
        ),
        process_options=ProcessCoupling(
            decay=True, diffusion=True, advection=True, sorption=True
        ),
        sim_setup=SimSetUp(
            timeloop=TimeLoop(
                processes=ProcessSettings(
                    nonlinear_solver_name="basic_picard",
                    convergence_type="PerComponentDeltaX",
                    norm_type="NORM2",
                    relative_tolerances=[1e-16] * (number_of_nuclides + 1),
                    time_discretization="BackwardEuler",
                ),
                time_stepping=FixedTimeStepping(
                    time_stepping_type="FixedTimeStepping",
                    t_initial=0.0,
                    t_end=COMPARISON_TIME_YEARS * SECONDS_PER_YEAR,
                    time_steps=[
                        1.0 * SECONDS_PER_YEAR,
                        10.0 * SECONDS_PER_YEAR,
                        100.0 * SECONDS_PER_YEAR,
                        1_000.0 * SECONDS_PER_YEAR,
                        10_000.0 * SECONDS_PER_YEAR,
                    ],
                    repeats=[100, 90, 90, 90, 90],
                ),
                output=OutputSettings(
                    repeats=[1, 1, 1, 1, 1],
                    each_steps=[100, 90, 90, 90, 90],
                ),
            ),
            non_linear_solver=NonLinearSolver(
                name="basic_picard",
                solver_type="Picard",
                max_iter=10,
                linear_solver="general_linear_solver",
            ),
            linear_solvers=LinearSolver(
                name=["general_linear_solver"] * 3,
                kind=["lis", "eigen", "petsc"],
                prefix=[None, None, "hc"],
                solver_type=["cg", "SparseLU", "bcgs"],
                precon_type=["jacobi", "ILUT", "bjacobi"],
                max_iteration_step=[20000, 10000, 20000],
                error_tolerance=[1e-16, 1e-14, 1e-8],
            ),
        ),
    )


def _run_ogs(tmp_path, geometry_data, rock_data, nuclide_data, sim_config):
    output_dir = tmp_path / "ogs_run"
    output_dir.mkdir()

    model = generate_ogsprj(
        output_dir=str(output_dir),
        geometry_database=geometry_data,
        rock_database=rock_data,
        nuclide_database=nuclide_data,
        sim_config=sim_config,
    )
    model.run_model(logfile=output_dir / "out.log", write_logs=True)

    return read_vtu_field(
        vtu_file_path=str(output_dir),
        pvd_name=f"{sim_config.project_name}.pvd",
        field_names=nuclide_data.nuclide_names,
        get_field_component_index=[0],
        sort_data=True,
        sort_by_index=2,
    )


def _analytical_concentrations(
    rock_data,
    rock_name,
    nuclide_data,
    nuclide_names,
    x_grids,
    c_init_value,
    c_inlet_value,
):
    rock_unit = rock_data.get(rock_name)
    porosity = rock_unit.get_property("porosity").value_for_run
    density = rock_unit.get_property("density").value_for_run

    diffusion_model = DiffusionModel()
    diffusion_coefficients = []
    sorption_coefficients = []
    for nuclide_name in nuclide_names:
        nuclide = nuclide_data.get(nuclide_name)
        diffusion_coefficients.append(
            default_pore_diffusion(rock_unit, nuclide, diffusion_model)
        )
        sorption_coefficients.append(
            nuclide.get_sorption_coefficient(rock_name).value_for_run
        )

    number_of_nuclides = len(nuclide_names)

    concentrations = Analytical_Sol().process_concentrations(
        nuclides=nuclide_names,
        c_init=np.full(number_of_nuclides, c_init_value),
        c_inlet=np.full(number_of_nuclides, c_inlet_value),
        t_interim=np.inf,
        t_timeseries=np.array([COMPARISON_TIME_YEARS * SECONDS_PER_YEAR]),
        x_grids=x_grids,
        decay=True,
        diffusion=True,
        advection=True,
        sorption=True,
        num_approx=True,
        scaling=False,
        branching_decay=False,
        diffusion_coefficients=diffusion_coefficients,
        media_porosities=[porosity],
        darcy_flux=[TARGET_DARCY_FLUX],
        media_densities=[density],
        sorption_coefficients=sorption_coefficients,
    )
    # shape: (1 time, number_of_nuclides, len(x_grids))
    return concentrations[0]


@pytest.mark.parametrize(
    ("scenario", "c_init_value", "c_inlet_value"),
    [
        pytest.param("injection", 0.0, 1.0, id="injection-init0-bc1"),
        pytest.param("washout", 1.0, 0.0, id="washout-init2-bc0"),
    ],
)
def test_1d_decay_chain_transport_matches_analytical_solution(
    tmp_path,
    geometry_data,
    rock_data,
    rock_name,
    units_1d,
    nuclide_data,
    nuclide_names,
    source,
    scenario,
    c_init_value,
    c_inlet_value,
):
    """OGS's numerical solution should agree with the closed-form one. For each nuclide, compare the two profiles via relative L2 norm
    and assert it stays small.
    """
    sim_config = _build_sim_config(
        rock_data,
        rock_name,
        units_1d,
        source,
        nuclide_names,
        c_init_value,
        c_inlet_value,
        project_name=f"test_claystone_decay_chain_{scenario}",
    )
    field_results = _run_ogs(
        tmp_path, geometry_data, rock_data, nuclide_data, sim_config
    )

    # OGS doesn't necessarily land exactly on COMPARISON_TIME_YEARS, so
    # snap to whichever output time is closest;
    target_time_s = COMPARISON_TIME_YEARS * SECONDS_PER_YEAR
    reference_entries = field_results[nuclide_names[0]]
    actual_time_s = min(
        {entry["t"] for entry in reference_entries},
        key=lambda t: abs(t - target_time_s),
    )

    # Distance from the source, derived from the OGS mesh's own node
    # z-coordinates at the chosen output time.
    at_time = [entry for entry in reference_entries if entry["t"] == actual_time_s]
    x_grids = np.sort(np.unique([abs(entry["z"] - source.loc) for entry in at_time]))

    analytical_profiles = _analytical_concentrations(
        rock_data,
        rock_name,
        nuclide_data,
        nuclide_names,
        x_grids,
        c_init_value,
        c_inlet_value,
    )

    for nuclide_index, nuclide_name in enumerate(nuclide_names):
        entries_at_time = [
            entry
            for entry in field_results[nuclide_name]
            if entry["t"] == actual_time_s
        ]

        # read_vtu_field already sorted these by z ascending (distance descending)
        # reverse to get the ascending order np.interp requires
        distances = np.array(
            [abs(entry["z"] - source.loc) for entry in entries_at_time]
        )[::-1]
        values = np.array([entry["val"] for entry in entries_at_time])[::-1]

        ogs_profile = np.clip(
            np.interp(x_grids, distances, values), CONCENTRATION_FLOOR, None
        )
        analytical_profile = np.clip(
            analytical_profiles[nuclide_index], CONCENTRATION_FLOOR, None
        )

        l2_error = np.linalg.norm(ogs_profile - analytical_profile)
        analytical_norm = np.linalg.norm(analytical_profile)
        relative_l2_error = (
            l2_error / analytical_norm if analytical_norm > 0 else l2_error
        )
        # Generous threshold against gross setup bugs
        assert relative_l2_error < 0.1, (
            f"[{scenario}] {nuclide_name} relative L2 error too high: "
            f"{relative_l2_error:.3e}"
        )
