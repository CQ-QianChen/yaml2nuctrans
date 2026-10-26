#!/usr/bin/env bash
# Usage:
#   python create_pseudo_data.py  
#   ./run_single.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output/single_run"
RESULTS_FILE="$OUTPUT_DIR/single_best_estimate_results.h5"

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
    --get_field_component_index "[2]" \
    --sort_by_index 2 \
    --run_mode single \
    --keep_vtu False \
    --path_to_save_results_hdf5_file "$RESULTS_FILE"
