from pathlib import Path

from ogstools.ogs6py import Project
from astropy import constants as const
from collections import deque
import numpy as np
import radioactivedecay as rd

from yaml2nuctrans.ogs_parameters.medium_properties import MediumProperties
from yaml2nuctrans.ogs_parameters.nuclide_properties import NuclidePropertiesCollection
from yaml2nuctrans.config.ogs_sim_config import SimulationConfig
from yaml2nuctrans.ogs_parameters.ogs_mesh_builder import MeshBuilder
from yaml2nuctrans.ogs_parameters.medium_ids import MediumIds
from yaml2nuctrans.ogs_parameters.phase_properties import PhaseProperties
from yaml2nuctrans.ogs_parameters.ogs_python_script import PythonBoundaryConditionScript
from yaml2nuctrans.primary_parameters.geometry import Units1D


def generate_ogsprj(
    output_dir,
    geometry_database,
    rock_database,
    nuclide_database,
    sim_config,
):
    """Create an OGS project using resolved model dataclasses."""
    
    project_name = sim_config.project_name

    phase_properties = PhaseProperties.from_rock_data(
    rock_data=rock_database,
    )

    medium_properties = MediumProperties.from_rock_data(
        rock_data=rock_database,
    )

    nuclide_properties = NuclidePropertiesCollection.from_databases(
        rock_data=rock_database,
        nuclide_data=nuclide_database
    )

    if sim_config.mesh_dimension == 1:
        units = Units1D.from_geometry_data(geometry_database)

    medium_ids = MediumIds.create_ids(
        sim_config=sim_config,
        units=units,
    )
    
    mesh_files = MeshBuilder.build(sim_config=sim_config, units=units, output_directory=output_dir)

    python_bc = PythonBoundaryConditionScript.from_config(config=sim_config,output_path=f"{output_dir}/python_DirichletBC.py")

    
    output_dir = Path(output_dir)

    model = Project(
        PROJECT_FILE=f"{project_name}.prj",
        output_dir=str(output_dir),
    )


    _add_meshes(
        model=model,
        meshes=mesh_files,
    )

    _add_python_script(
        model=model,
        python_bc=python_bc,
    )

    _add_process(
        model=model,
        meshes=mesh_files,
        sim_config=sim_config,
        nuclide_database = nuclide_database
    )

    # ---- add numerical_stabilization ---- #
    model.add_block(
        blocktag="numerical_stabilization",
        parent_xpath="./processes/process",
        taglist=["type"],
        textlist=["FluxCorrectedTransport"],
    )

    _add_media(
        model=model,
        medium_ids=medium_ids,
        medium_properties=medium_properties,
        phase_properties=phase_properties,
        nuclide_properties=nuclide_properties,
        sim_config=sim_config
    )

    _add_time_loop(
    model=model,
    sim_config=sim_config,
    )

    _add_chemical_system(
        model=model,
        meshes=mesh_files,
        sim_config=sim_config,
        nuclide_database = nuclide_database
    )

    _add_initial_and_boundary_conditions(
        model=model,
        meshes=mesh_files,
        sim_config=sim_config,
        nuclide_database = nuclide_database
    )

    _add_solvers(
        model=model,
        sim_config=sim_config,
    )

    model.write_input(
        prjfile_path=str(
            output_dir
            / f"{project_name}.prj"
        )
    )

    return model

def _add_meshes(
    model,
    meshes,
) -> None:
    """Add all required OGS meshes from MeshFiles1D."""
   
    mesh_directory = Path(meshes.file_path)

    mesh_names = (
        meshes.domain,
        meshes.top,
        meshes.bottom,
        meshes.reactive_domain,
    )

    for mesh_name in mesh_names:
        mesh_path = mesh_directory / f"{mesh_name}.vtu"

        model.mesh.add_mesh(
            filename=str(mesh_path),
        )

def _add_python_script(
    model: Project,
    python_bc: PythonBoundaryConditionScript,
) -> None:
    """Register the generated or user-provided Python BC script."""
    
    if python_bc.output_path is None:
        return

    python_bc_path = Path(python_bc.output_path)

    model.python_script.set_pyscript(
        filename=str(python_bc_path),
    )

def _add_process(
    model,
    meshes,
    sim_config,
    nuclide_database,
):
    """Add the ComponentTransport process and its variables."""

    advection = sim_config.process_options.advection
    body_force = " ".join(
        ["0"] * meshes.dimension
    )

    if advection:
        body_force = " ".join(
            ["0"] * (meshes.dimension - 1)
            + [f"-{const.g0.value}"]
        )

    model.processes.set_process(
        name="hc",
        type="ComponentTransport",
        integration_order="2",
        specific_body_force=body_force,
    )

    for process_variable in sim_config.process_variables:
        if process_variable == 'concentration':
            for nuclide in nuclide_database.nuclide_names:
                model.processes.add_process_variable(
                                process_variable=process_variable,
                                process_variable_name=nuclide,
                            )
        else:
            model.processes.add_process_variable(
                    process_variable=process_variable, process_variable_name=process_variable
                )


    secondary_variables = sim_config.secondary_variables
    if secondary_variables is not None:
        for secondary_variable in secondary_variables:
            if secondary_variable == 'Flux':
                for nuclide in nuclide_database.nuclide_names:
                    model.processes.add_secondary_variable(internal_name=nuclide+secondary_variable, output_name=nuclide+secondary_variable)
            else:
                model.processes.add_secondary_variable(internal_name=secondary_variable, output_name=secondary_variable)

def _add_media(
    model: Project,
    medium_ids: MediumIds,
    medium_properties: MediumProperties,
    phase_properties: PhaseProperties,
    nuclide_properties: NuclidePropertiesCollection,
    sim_config: SimulationConfig
) -> None:
    """Add OGS media, phases, and component properties."""


    for medium_id, rock_name in medium_ids.by_id.items():
        _add_medium_properties(
            model=model,
            medium_id=medium_id,
            rock_name=rock_name,
            medium_properties=medium_properties,
        )

        _add_phase_properties(
            model=model,
            medium_id=medium_id,
            rock_name=rock_name,
            phase_properties=phase_properties,
        )

        _add_nuclide_properties(
            model=model,
            medium_id=medium_id,
            rock_name=rock_name,
            nuclide_properties=nuclide_properties,
            sim_config=sim_config
        )

def _add_medium_properties(
    model: Project,
    medium_id: str,
    rock_name: str,
    medium_properties: MediumProperties,
) -> None:
    """Add medium-level properties for one geological unit."""
    medium = medium_properties.get(rock_name)

    for property_name, property_data in (
        medium.properties.items()
    ):
        _add_ogs_property(
            model=model,
            medium_id=medium_id,
            name=property_name,
            property_type=property_data.property_type,
            value=property_data.value,
        )

def _add_phase_properties(
    model: Project,
    medium_id: str,
    rock_name: str,
    phase_properties: PhaseProperties,
) -> None:
    """Add phase properties, for example aqueous density and viscosity."""
    phases = phase_properties.by_rock_name[rock_name]

    for phase_type, phase in phases.items():
        for property_name, property_data in (
            phase.properties.items()
        ):
            _add_ogs_property(
                model=model,
                medium_id=medium_id,
                phase_type=phase_type,
                name=property_name,
                property_type=property_data.property_type,
                value=property_data.value,
            )

def _add_nuclide_properties(
    model: Project,
    medium_id: str,
    rock_name: str,
    nuclide_properties: NuclidePropertiesCollection,
    sim_config: SimulationConfig
) -> None:
    """Add component properties for every phase and nuclide."""

    phases = nuclide_properties.by_rock[rock_name]
    diffusion = sim_config.process_options.diffusion
    sorption = sim_config.process_options.sorption

    for phase_type, nuclides in phases.items():
        for nuclide_name, nuclide_data in nuclides.items():
            for property_name, property_data in (
                nuclide_data.properties.items()
            ):
                value = property_data.value

                if (
                    property_name == "pore_diffusion"
                    and not diffusion
                ):
                    value = 0.0

                if (
                    property_name == "retardation_factor"
                    and not sorption
                ):
                    value = 1.0

                _add_ogs_property(
                    model=model,
                    medium_id=medium_id,
                    phase_type=phase_type,
                    component_name=nuclide_name,
                    name=property_name,
                    property_type=property_data.property_type,
                    value=value,
                )

def _add_ogs_property(
    model: Project,
    medium_id: str,
    name: str,
    property_type: str,
    value: float | str,
    phase_type: str | None = None,
    component_name: str | None = None,
) -> None:
    """Add one Constant or Function property to an OGS medium."""
    property_arguments = {
        "medium_id": medium_id,
        "name": name,
        "type": property_type,
    }

    if phase_type is not None:
        property_arguments["phase_type"] = phase_type

    if component_name is not None:
        property_arguments["component_name"] = component_name

    if property_type == "Constant":
        model.media.add_property(
            **property_arguments,
            value=value,
        )
        return

    if property_type == "Function":
        model.media.add_property(
            **property_arguments,
            value="",
            expression=value,
        )
        return

    raise NotImplementedError(
        f"OGS property type '{property_type}' is not supported."
    )

def _add_time_loop(
    model,
    sim_config,
) -> None:
    """Add OGS time-loop, time-stepping, and output settings."""
    processes = sim_config.sim_setup.timeloop.processes
    time_stepping = sim_config.sim_setup.timeloop.time_stepping
    output = sim_config.sim_setup.timeloop.output

    model.time_loop.add_process(
        process="hc",
        nonlinear_solver_name=(
            processes.nonlinear_solver_name
        ),
        convergence_type=processes.convergence_type,
        norm_type=processes.norm_type,
        reltols=" ".join(map(str, processes.relative_tolerances)),
        time_discretization=processes.time_discretization,
    )

    if time_stepping.time_stepping_type == "FixedTimeStepping":

        model.time_loop.set_stepping(
            process="hc",
            type=time_stepping.time_stepping_type,
            t_initial=time_stepping.t_initial,
            t_end=time_stepping.t_end,
            repeat=time_stepping.repeats,
            delta_t=time_stepping.time_steps,
        )
    elif time_stepping.time_stepping_type == "IterationNumberBasedTimeStepping":
        model.time_loop.set_stepping(
            process="hc",
            type=time_stepping.time_stepping_type,
            t_initial=time_stepping.t_initial,
            initial_dt=time_stepping.initial_dt,
            minimum_dt=time_stepping.minimum_dt,
            maximum_dt=time_stepping.maximum_dt,
            number_iterations=time_stepping.number_iterations,
            multiplier=time_stepping.multiplier,
        )

    else:
        raise NotImplementedError(
            f"Time step method '{time_stepping.time_stepping_type}' is not implemented yet."
        )

    model.time_loop.add_output(
        type="VTK",
        prefix=sim_config.project_name,
        suffix="_ts_{:timestep}_t_{:time}",
        repeat=output.repeats,
        each_steps=output.each_steps,
        variables=[],
    )

def _add_chemical_system(
    model,
    meshes,
    sim_config,
    nuclide_database,
) -> None:
    """Add radioactive-decay reactions when decay is enabled."""


    if not sim_config.process_options.decay:
        return
    
    nuclides = nuclide_database.nuclide_names
    number_of_nuclides = len(nuclides)

    nuclide_half_lives = [
        rd.Nuclide(nuclide).half_life()
        for nuclide in nuclides
    ]

    decay_rates = (
        np.log(2)
        / np.array(nuclide_half_lives)
    )

    base_coefficients = (
        ["-1", "1"]
        + ["0"] * (number_of_nuclides - 2)
    )

    model.add_block(
        blocktag="chemical_system",
        block_attrib={
            "chemical_solver": "SelfContained",
        },
        parent_xpath=".",
        taglist=[
            "mesh",
            "linear_solver",
            "number_of_components",
        ],
        textlist=[
            meshes.reactive_domain,
            "general_linear_solver",
            str(number_of_nuclides),
        ],
    )

    model.add_element(
        tag="chemical_reactions",
        parent_xpath="./chemical_system",
    )

    for index, decay_rate in enumerate(decay_rates):
        coefficients = deque(base_coefficients)
        coefficients.rotate(index)

        if index == number_of_nuclides - 1:
            coefficients = ["0"] * index + ["-1"]

        model.add_block(
            blocktag="chemical_reaction",
            parent_xpath=(
                "./chemical_system/chemical_reactions"
            ),
            taglist=[
                "stoichiometric_coefficients",
                "reaction_type",
                "first_order_rate_constant",
            ],
            textlist=[
                " ".join(coefficients),
                "FirstOrderReaction",
                str(decay_rate),
            ],
        )

def _add_initial_and_boundary_conditions(
    model,
    meshes,
    sim_config,
    nuclide_database
) -> None:
    """Add pressure and nuclide initial and boundary conditions."""
    _add_pressure_conditions(
        model=model,
        meshes= meshes,
        sim_config=sim_config
    )

    _add_nuclide_conditions(
        model=model,
        sim_config=sim_config,
        nuclide_database = nuclide_database
    )

def _add_pressure_conditions(
    model,
    meshes,
    sim_config
) -> None:
    """Add pressure initial condition and pressure boundaries."""
    pressure = sim_config.pressure_setup
    advection = sim_config.process_options.advection

    model.process_variables.set_ic(
        process_variable_name="pressure",
        components="1",
        order="1",
        initial_condition="p_init",
    )

    if not advection:
        model.parameters.add_parameter(
            name="p_init",
            type="Constant",
            value=0.0,
        )
        return

    model.parameters.add_parameter(
        name="p_init",
        type="Constant",
        value=pressure.pressure_initial,
    )

    model.parameters.add_parameter(
        name="p_top",
        type="Constant",
        value=pressure.pressure_top,
    )

    model.parameters.add_parameter(
        name="p_bottom",
        type="Constant",
        value=pressure.pressure_bottom,
    )

    model.process_variables.add_bc(
        process_variable_name="pressure",
        mesh=meshes.top,
        type="Dirichlet",
        parameter="p_top",
    )

    model.process_variables.add_bc(
        process_variable_name="pressure",
        mesh=meshes.bottom,
        type="Dirichlet",
        parameter="p_bottom",
    )

def _add_nuclide_conditions(
    model,
    sim_config,
    nuclide_database
) -> None:
    """Add initial and boundary conditions for all nuclides."""
    initial = sim_config.initial_condition_nuclide
    boundary = sim_config.boundary_condition_nuclide
    nuclides = nuclide_database.nuclide_names

    for (
        nuclide,
        initial_type,
        initial_value,
        boundary_type,
        boundary_mesh,
        boundary_value,
    ) in zip(
        nuclides,
        initial.types,
        initial.values,
        boundary.types,
        boundary.meshes,
        boundary.values,
        strict=True,
    ):
        initial_parameter_name = f"c_init_{nuclide}"

        _add_parameter(
            model=model,
            name=initial_parameter_name,
            property_type=initial_type,
            value=initial_value,
        )

        model.process_variables.set_ic(
            process_variable_name=nuclide,
            components="1",
            order="1",
            initial_condition=initial_parameter_name,
        )

        _add_nuclide_boundary_condition(
            model=model,
            nuclide=nuclide,
            boundary_type=boundary_type,
            boundary_mesh=boundary_mesh,
            boundary_value=boundary_value,
        )

def _add_nuclide_boundary_condition(
    model,
    nuclide: str,
    boundary_type: str,
    boundary_mesh: str,
    boundary_value: float | str,
) -> None:
    """Add one nuclide boundary condition."""
    if boundary_type in {"Dirichlet", "Neumann"}:
        parameter_name = f"c_bc_{nuclide}"

        property_type = "Constant"

        if isinstance(boundary_value, str):
            property_type = "Function"

        _add_parameter(
            model=model,
            name=parameter_name,
            property_type=property_type,
            value=boundary_value,
        )

        model.process_variables.add_bc(
            process_variable_name=nuclide,
            mesh=boundary_mesh,
            type=boundary_type,
            parameter=parameter_name,
        )

        return

    if boundary_type == "Python":
        model.process_variables.add_bc(
            process_variable_name=nuclide,
            mesh=boundary_mesh,
            type="Python",
            bc_object=boundary_value,
        )

        return

    raise NotImplementedError(
        f"Boundary condition type '{boundary_type}' "
        "is not supported."
    )

def _add_parameter(
    model,
    name: str,
    property_type: str,
    value: float | str,
) -> None:
    """Add an OGS Constant or Function parameter."""
    if property_type == "Constant":
        model.parameters.add_parameter(
            name=name,
            type="Constant",
            value=value,
        )
        return

    if property_type == "Function":
        model.parameters.add_parameter(
            name=name,
            type="Function",
            expression=value,
        )
        return

    raise NotImplementedError(
        f"Parameter type '{property_type}' is not supported."
    )

def _add_solvers(
    model,
    sim_config,
) -> None:
    """Add nonlinear and linear solver definitions."""
    nonlinear_solver = sim_config.sim_setup.non_linear_solver

    model.nonlinear_solvers.add_non_lin_solver(
        name=nonlinear_solver.name,
        type=nonlinear_solver.solver_type,
        max_iter=nonlinear_solver.max_iter,
        linear_solver=nonlinear_solver.linear_solver,
    )

    linear_solvers = sim_config.sim_setup.linear_solvers
    number_of_linsovers = len(linear_solvers.name)

    for i in range(number_of_linsovers):
        model.linear_solvers.add_lin_solver(
            name=linear_solvers.name[i],
            kind=linear_solvers.kind[i],
            prefix=linear_solvers.prefix[i],
            solver_type=linear_solvers.solver_type[i],
            precon_type=linear_solvers.precon_type[i],
            max_iteration_step=linear_solvers.max_iteration_step[i],
            error_tolerance=linear_solvers.error_tolerance[i],
        )