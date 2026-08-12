import shutil
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import os
import sys

from yaml2nuctrans.primary_parameters.rock_database import RockData
from yaml2nuctrans.primary_parameters.nuclide_database import NuclideData
from yaml2nuctrans.config.ogs_sim_config import SimulationConfig
from yaml2nuctrans.primary_parameters.geometry import GeometryData
from yaml2nuctrans.read_results.read_ogs_model_results import read_vtu_field, read_sim_results, build_result_template, merge_sample_results, save_results_to_hdf5
from yaml2nuctrans.config.generate_ogs_prj import generate_ogsprj
from yaml2nuctrans.sampling.apply_sampled_data import count_samples, load_sampled_data, apply_sample


def _run_single_sample_worker(
    i,
    output_directory,
    rock_data_folder_path,
    site_folder_path,
    geometry_folder_path,
    emitted_energy_folder_path,
    species_type_folder_path,
    sorption_data_folder_path,
    nuclide_water_diffusivity_folder_path,
    model_config_path,
    uncertain_parameters,
    get_field_component_index,
    sort_by_index,
    keep_vtu=False,
):
    """
    Run one ensemble sample i
    """

    # load input data (per sample, so no pickling required)
    rock_database = RockData.from_folder(
        rock_data_folder=rock_data_folder_path,
        site_data_folder=site_folder_path,
    )
    nuclide_database = NuclideData.from_yaml_files(
        emitted_energy_folder=emitted_energy_folder_path,
        species_type_folder=species_type_folder_path,
        sorption_data_folder=sorption_data_folder_path,
        nuclide_water_diffusivity_folder=nuclide_water_diffusivity_folder_path,
    )
    sim_config = SimulationConfig.from_yaml(filename=model_config_path)
    geometry_database = GeometryData.from_folder(
        geometry_data_path=geometry_folder_path
    )

    # apply sampled parameters for this sample
    apply_sample(
        i=i,
        uncertain_parameters=uncertain_parameters,
        geometry_database=geometry_database,
        rock_database=rock_database,
        nuclide_database=nuclide_database,
    )

    # separate output folder per sample so results are not overwritten
    sample_output_dir = str(Path(output_directory) / f"sample_{i:04d}")

    model_for_run = generate_ogsprj(
        output_dir=sample_output_dir,
        geometry_database=geometry_database,
        rock_database=rock_database,
        nuclide_database=nuclide_database,
        sim_config=sim_config,
    )
    model_for_run.run_model(logfile=Path(sample_output_dir)/'out.log' ,write_logs=True)

    # read QoI results for this sample
    qoi_results = read_sim_results(
        sample_output_dir,
        sim_config,
        nuclide_database,
        get_field_component_index,
        sort_by_index,
    )

    # delete the sample's output folder
    if not keep_vtu:
        shutil.rmtree(sample_output_dir, ignore_errors=True)

    return qoi_results


def ogs_model(
    output_directory,
    rock_data_folder_path,
    site_folder_path,
    geometry_folder_path,
    emitted_energy_folder_path,
    species_type_folder_path,
    sorption_data_folder_path,
    nuclide_water_diffusivity_folder_path,
    model_config_path,
    sampled_data_file_path=None,
    get_field_component_index=[0, 1, 2],
    sort_by_index=2,
    run_mode = 'ensemble',
    parallel=True,
    n_jobs=-1,
    keep_vtu=False,
    **kwargs
):

    # load input data for single-run mode
    rock_database = RockData.from_folder(
        rock_data_folder=rock_data_folder_path,
        site_data_folder=site_folder_path,
    )
    nuclide_database = NuclideData.from_yaml_files(
        emitted_energy_folder=emitted_energy_folder_path,
        species_type_folder=species_type_folder_path,
        sorption_data_folder=sorption_data_folder_path,
        nuclide_water_diffusivity_folder=nuclide_water_diffusivity_folder_path,
    )

    sim_config = SimulationConfig.from_yaml(filename=model_config_path)
    geometry_database = GeometryData.from_folder(
        geometry_data_path=geometry_folder_path
    )

    if run_mode == "single":
        # run ogs simulation
        output_dir = str(Path(output_directory) / "single_best_estimate")
        model_for_run = generate_ogsprj(
            output_dir=output_dir,
            geometry_database=geometry_database,
            rock_database=rock_database,
            nuclide_database=nuclide_database,
            sim_config=sim_config,
        )
        model_for_run.run_model(logfile=Path(sample_output_dir)/'out.log' ,write_logs=True)
        results = read_sim_results(
            output_dir, sim_config, nuclide_database,
            get_field_component_index, sort_by_index,
        )
        if not keep_vtu:
            shutil.rmtree(output_dir, ignore_errors=True)
        return results

    elif run_mode == "ensemble":
        # load samples
        sampled_data = load_sampled_data(sampled_data_file_path)
        uncertain_parameters = sampled_data["uncertain_parameters"]
        n_samples = count_samples(uncertain_parameters)

        all_run_results = build_result_template(sim_config, nuclide_database)

        if parallel:
            # parallel ensemble run
            results_gen = Parallel(n_jobs=n_jobs, return_as="generator")( 
                delayed(_run_single_sample_worker)(
                    i,
                    output_directory,
                    rock_data_folder_path,
                    site_folder_path,
                    geometry_folder_path,
                    emitted_energy_folder_path,
                    species_type_folder_path,
                    sorption_data_folder_path,
                    nuclide_water_diffusivity_folder_path,
                    model_config_path,
                    uncertain_parameters,
                    get_field_component_index,
                    sort_by_index,
                    keep_vtu,
                )
                for i in range(n_samples)
            )

            # wrap result generator with tqdm to track completed samples
            for i, qoi_results in enumerate(
                tqdm(results_gen, total=n_samples, desc="OGS samples", unit="sample")
            ):
                all_run_results = merge_sample_results(
                    all_run_results, qoi_results, sample_index=i
                )

        else:
            # sequential ensemble run 
            for i in tqdm(
                range(n_samples), desc="OGS samples", unit="sample"
            ):
                apply_sample(
                    i=i,
                    uncertain_parameters=uncertain_parameters,
                    geometry_database=geometry_database,
                    rock_database=rock_database,
                    nuclide_database=nuclide_database,
                )

                sample_output_dir = str(Path(output_directory) / f"sample_{i:04d}")

                model_for_run = generate_ogsprj(
                    output_dir=sample_output_dir,
                    geometry_database=geometry_database,
                    rock_database=rock_database,
                    nuclide_database=nuclide_database,
                    sim_config=sim_config,
                )
                model_for_run.run_model(logfile=Path(sample_output_dir)/'out.log' ,write_logs=True)

                qoi_results = read_sim_results(
                    sample_output_dir,
                    sim_config,
                    nuclide_database,
                    get_field_component_index,
                    sort_by_index,
                )
                all_run_results = merge_sample_results(
                    all_run_results, qoi_results, sample_index=i
                )

                # delete the sample's output folder
                if not keep_vtu:
                    shutil.rmtree(sample_output_dir, ignore_errors=True)

        return all_run_results

    else:
        raise ValueError(
            f"Invalid run mode: {sim_config.run.get('mode')}. Must be 'single' or 'ensemble'."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run OGS model (single or ensemble) from YAML/folder inputs.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--output_directory", required=True,
                        help="Directory where model outputs are written.")
    parser.add_argument("--rock_data_folder_path", required=True,
                        help="Folder with rock property data.")
    parser.add_argument("--site_folder_path", required=True,
                        help="Folder with site-specific data.")
    parser.add_argument("--geometry_folder_path", required=True,
                        help="Folder with geometry data (rock interfaces etc.).")
    parser.add_argument("--emitted_energy_folder_path", required=True,
                        help="Folder with nuclide emitted energy data.")
    parser.add_argument("--species_type_folder_path", required=True,
                        help="Folder with nuclide species type data.")
    parser.add_argument("--sorption_data_folder_path", required=True,
                        help="Folder with nuclide sorption coefficient data.")
    parser.add_argument("--nuclide_water_diffusivity_folder_path", required=True,
                        help="Folder with nuclide water diffusivity data.")
    parser.add_argument("--model_config_path", required=True,
                        help="Path to the YAML simulation config file.")

    parser.add_argument("--sampled_data_file_path", default=None,
                        help="HDF5 file with sampled uncertain parameters (required for ensemble mode).")
    parser.add_argument("--get_field_component_index", default='[0, 1, 2]',
                        help="Which vector components of the fields to extract (e.g. [2] or [0, 2] or [0, 1, 2].")
    parser.add_argument("--sort_by_index", type=int, default=2,
                        help="Coordinate index used to sort the results (0=x, 1=y, 2=z).")
    parser.add_argument("--run_mode", default='ensemble',
                        help="Run mode: 'single' for single best estimate, 'ensemble' for ensemble run with sampled parameters.")
    parser.add_argument("--parallel", default='True',
                        help="Run ensemble samples in parallel using joblib.")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel worker processes (-1 = all CPUs). Only used with --parallel.")
    parser.add_argument("--keep_vtu", default="False",
                        help="Keep per-sample output folders (VTU/PVD files) after reading results. "
                             "By default they are deleted once results are written into Python dictionary.")
    parser.add_argument("--path_to_save_results_hdf5_file", default=None,
                        help="Path to save the results dictionary to a HDF5 file.")
    parser.add_argument("--save_sampled_data", default="True",
                        help="Whether to save the sampled data into the HDF5 results file. "
                             "By default it is saved together with the simulationresults.")

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    results = ogs_model(
        output_directory=args.output_directory,
        rock_data_folder_path=args.rock_data_folder_path,
        site_folder_path=args.site_folder_path,
        geometry_folder_path=args.geometry_folder_path,
        emitted_energy_folder_path=args.emitted_energy_folder_path,
        species_type_folder_path=args.species_type_folder_path,
        sorption_data_folder_path=args.sorption_data_folder_path,
        nuclide_water_diffusivity_folder_path=args.nuclide_water_diffusivity_folder_path,
        model_config_path=args.model_config_path,
        sampled_data_file_path=args.sampled_data_file_path,
        get_field_component_index=eval(args.get_field_component_index),
        sort_by_index=args.sort_by_index,
        run_mode=args.run_mode,
        parallel=eval(args.parallel),
        n_jobs=args.n_jobs,
        keep_vtu=eval(args.keep_vtu)
    )

    output_dir = Path(args.output_directory)

    output_dir.mkdir(parents=True, exist_ok=True)

    # save results into one HDF5 file.
    sampled_data = None
    save_sampled_data = eval(args.save_sampled_data)
    

    if args.run_mode == "ensemble":
        sampled_data_file_path = args.sampled_data_file_path
        if save_sampled_data:
            save_results_name = output_dir / f"ensemble_results_with_{Path(sampled_data_file_path).stem}.h5"
            sampled_data = load_sampled_data(sampled_data_file_path)
        else:
            save_results_name = output_dir / f"ensemble_results_using_{Path(sampled_data_file_path).stem}.h5"
    elif args.run_mode == "single":
        save_results_name = output_dir / f"single_best_estimate_results.h5"
    else:
        raise ValueError(
            f"Invalid run mode: {args.run_mode}. Must be 'single' or 'ensemble'."
        )

    path_to_save_results = args.path_to_save_results_hdf5_file
    if (path_to_save_results is not None) and (path_to_save_results != "default"):
        save_results_name = Path(path_to_save_results)
        save_results_name.parent.mkdir(parents=True, exist_ok=True)
    
    save_results_to_hdf5(results, save_results_name, sampled_data=sampled_data, save_sampled_data=save_sampled_data)

if __name__ == "__main__":
    main()