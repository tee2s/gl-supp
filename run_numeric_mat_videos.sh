#!/usr/bin/env bash
set -euo pipefail

# gets the name of the directory where the script is stored, no matter from where it is called
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/work/users/t/i/tis/data/jhu_spatiotemporal/fastdata260421_no_compression"
OUT_DIR="/work/users/t/i/tis/data/jhu_spatiotemporal/beamformed_videos/skip8"

shopt -s nullglob

for mat_file in "$DATA_DIR"/*.mat; do
    filename="$(basename -- "$mat_file")"

    if [[ ! "$filename" =~ ^[0-9]+\.mat$ ]]; then
        continue
    fi

    echo "Processing $mat_file"
    python "$SCRIPT_DIR/beamform_torch.py" "$mat_file" \
        --start 0 \
        --stop 100 \
        --step 1 \
        --channel-skip 8 \
        --device cuda \
        --out-format video \
        --out-dir "$OUT_DIR"
done
