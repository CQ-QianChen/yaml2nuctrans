#!/usr/bin/env bash
# Usage:
#   python create_pseudo_data.py   
#   ./make_ensemble_samples.sh
#   ./run_ensemble.sh              # generates the ensemble samples itself if missing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output/ensemble_run"
RESULTS_FILE="$OUTPUT_DIR/ensemble_results.h5"
SAMPLED_DATA_DIR="$DATA_DIR/sampled_data"

# load sampled-data file; generate it if missing.
shopt -s nullglob
sampled_data_files=("$SAMPLED_DATA_DIR"/*.h5)
shopt -u nullglob
if [[ ${#sampled_data_files[@]} -eq 0 ]]; then
    "$SCRIPT_DIR/make_ensemble_samples.sh"
    shopt -s nullglob
    sampled_data_files=("$SAMPLED_DATA_DIR"/*.h5)
    shopt -u nullglob
fi
SAMPLED_DATA_FILE="${sampled_data_files[0]}"

python -m yaml2nuctrans.get_model.ogs_model \
    --output_directory "$OUTPUT_DIR" \
    --rock_data_folder_path "$DATA_DIR/rock_data" \
    --site_folder_path "$DATA_DIR/site_data" \
    --geometry_folder_path "$DATA_DIR/geometry" \
    --emitted_energy_folder_path "$DATA_DIR/nuclide_emitted_energy_data" \
    --species_type_folder_path "$DATA_DIR/nuclide_species_data" \
    --sorption_data_folder_path "$DATA_DIR/nuclide_sorption_data" \
    --nuclide_water_diffusivity_folder_path "$DATA_DIR/nuclide_water_diffusivity_data" \
    --model_config_path "$SCRIPT_DIR/config_point_source.yaml" \
    --sampled_data_file_path "$SAMPLED_DATA_FILE" \
    --get_field_component_index "[2]" \
    --sort_by_index 2 \
    --run_mode ensemble \
    --parallel True \
    --n_jobs 4 \
    --keep_vtu False \
    --path_to_save_results_hdf5_file "$RESULTS_FILE" \
    --save_sampled_data True
