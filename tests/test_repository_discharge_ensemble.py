import importlib.util
import shutil
from pathlib import Path

import pytest

from yaml2nuctrans.config.generate_ogs_prj import generate_ogsprj
from yaml2nuctrans.config.ogs_sim_config import SimulationConfig
from yaml2nuctrans.ogs_parameters.medium_ids import MediumIds
from yaml2nuctrans.ogs_parameters.medium_properties import MediumProperties
from yaml2nuctrans.post_processing.repository_discharge import (
    DOMAIN_ZONE,
    compute_zone_substance_amounts,
    ewg_zones_from_host_rock,
)
from yaml2nuctrans.primary_parameters.geometry import GeometryData, Units1D
from yaml2nuctrans.primary_parameters.nuclide_database import NuclideData
from yaml2nuctrans.primary_parameters.rock_database import RockData
from yaml2nuctrans.read_results.read_ogs_model_results import merge_sample_results

pytestmark = pytest.mark.skipif(
    shutil.which("ogs") is None,
    reason="OGS executable not available on PATH",
)

EXAMPLE_DIR = (
    Path(__file__).parent.parent / "examples" / "yaml_pseudo_decay_chain_transport"
)
HOST_ROCK_NAME = "Pseudo_Claystone"

# Two sampled Pseudo_Claystone porosities, the YAML's own `inventory_mol` stays fixed across samples.
ENSEMBLE_POROSITIES = [0.22, 0.30]


def _write_pseudo_dataset(data_dir):
    """Run the example's `create_pseudo_data.py`."""
    spec = importlib.util.spec_from_file_location(
        "create_pseudo_data", EXAMPLE_DIR / "create_pseudo_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = data_dir
    module.main()
    return module


def test_ensemble_initial_domain_substance_amount_matches_inventory_mol(tmp_path):
    """Run the Cm-247 decay-chain pseudo example, build from YAML files."""
    data_dir = tmp_path / "data"
    _write_pseudo_dataset(data_dir)

    rock_database = RockData.from_folder(
        rock_data_folder=data_dir / "rock_data",
        site_data_folder=data_dir / "site_data",
    )
    nuclide_database = NuclideData.from_yaml_files(
        emitted_energy_folder=data_dir / "nuclide_emitted_energy_data",
        species_type_folder=data_dir / "nuclide_species_data",
        sorption_data_folder=data_dir / "nuclide_sorption_data",
        nuclide_water_diffusivity_folder=data_dir / "nuclide_water_diffusivity_data",
    )
    geometry_database = GeometryData.from_folder(
        geometry_data_path=data_dir / "geometry"
    )
    units = Units1D.from_geometry_data(geometry_database)

    inventory_mol = None
    all_run_results = {
        f"{nuclide_name}_substance_amount": {}
        for nuclide_name in nuclide_database.nuclide_names
    }

    for sample_index, porosity in enumerate(ENSEMBLE_POROSITIES):
        # Reloaded fresh each sample
        rock_database.get(HOST_ROCK_NAME).get_property("porosity").run_value = porosity
        sim_config = SimulationConfig.from_yaml(EXAMPLE_DIR / "config_line_source.yaml")
        inventory_mol = (
            sim_config.initial_condition_nuclide.source_concentration.inventory_mol
        )

        output_dir = tmp_path / f"sample_{sample_index:04d}"
        model = generate_ogsprj(
            output_dir=str(output_dir),
            geometry_database=geometry_database,
            rock_database=rock_database,
            nuclide_database=nuclide_database,
            sim_config=sim_config,
        )
        model.run_model(logfile=output_dir / "out.log", write_logs=True)

        medium_ids = MediumIds.create_ids(sim_config=sim_config, units=units)
        domain_mesh, zones = ewg_zones_from_host_rock(
            domain_mesh=output_dir / f"{sim_config.meshes.domain}.vtu",
            host_rock=sim_config.repository_discharge.host_rock,
            medium_ids=medium_ids,
        )
        medium_properties = MediumProperties.from_rock_data(rock_data=rock_database)
        porosity_by_material_id = {
            int(medium_id): medium_properties.get(layer_name).porosity.value
            for medium_id, layer_name in medium_ids.by_id.items()
        }

        substance_amounts = compute_zone_substance_amounts(
            result_pvd_path=output_dir / f"{sim_config.project_name}.pvd",
            domain_mesh=domain_mesh,
            nuclide_names=nuclide_database.nuclide_names,
            porosity_by_material_id=porosity_by_material_id,
            area_repository=sim_config.initial_condition_nuclide.source_concentration.area_repository,
            zones=zones,
        )
        all_run_results = merge_sample_results(
            all_run_results, substance_amounts, sample_index=sample_index
        )

    for nuclide_name, inventory in zip(nuclide_database.nuclide_names, inventory_mol):
        zone_series = all_run_results[f"{nuclide_name}_substance_amount"]
        times = sorted({t for _, t in zone_series})

        initial_amounts = zone_series[(DOMAIN_ZONE, times[0])]
        assert len(initial_amounts) == len(ENSEMBLE_POROSITIES)
        for amount in initial_amounts:
            assert amount == pytest.approx(inventory, rel=1e-6)
