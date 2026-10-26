#!/usr/bin/env bash
# Usage:
#   python create_pseudo_data.py  
#   ./make_ensemble_samples.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

python -m yaml2nuctrans.sampling.sampling_func \
    --config "$SCRIPT_DIR/sample_config.yaml" \
    --sample_size 8 \
    --sampling_method sobol \
    --seed 21 \
    --path_to_geometry_data "$DATA_DIR/geometry" \
    --path_to_rock_data "$DATA_DIR/rock_data" \
    --path_to_sorption_data "$DATA_DIR/nuclide_sorption_data" \
    --path_to_diffusivity_data "$DATA_DIR/nuclide_water_diffusivity_data" \
    --path_to_save_sampled_data "$DATA_DIR/sampled_data" \
    --save_file_type HDF5
