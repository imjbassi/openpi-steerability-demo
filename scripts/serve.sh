#!/usr/bin/env bash
# Start the openpi policy server with the pi05_libero checkpoint, after
# checking that enough VRAM is actually free (openpi documents > 8 GB for
# inference, and JAX preallocates a fraction of TOTAL VRAM by default).
#
# First run downloads the checkpoint from gs://openpi-assets to
# ~/.cache/openpi (several GB).
set -euo pipefail
cd "$(dirname "$0")/.."

MIN_FREE_MB=8500

if command -v nvidia-smi >/dev/null; then
    read -r FREE_MB TOTAL_MB < <(nvidia-smi --query-gpu=memory.free,memory.total \
        --format=csv,noheader,nounits | head -1 | tr -d ',')
    echo "GPU memory: ${FREE_MB} MiB free of ${TOTAL_MB} MiB"
    if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
        echo "ERROR: only ${FREE_MB} MiB free, but pi05 inference needs > 8 GB." >&2
        echo "Close whatever is holding VRAM (check nvidia-smi) and retry." >&2
        exit 1
    fi
    # Let JAX use most of what is actually free, not 75% of total.
    FRACTION=$(awk -v f="$FREE_MB" -v t="$TOTAL_MB" \
        'BEGIN { x = (f - 512) / t; if (x > 0.9) x = 0.9; printf "%.2f", x }')
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$FRACTION"
    echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION"
else
    echo "WARNING: nvidia-smi not found; starting anyway" >&2
fi

cd vendor/openpi
exec uv run scripts/serve_policy.py --env LIBERO "$@"
