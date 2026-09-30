#!/usr/bin/env bash
# Clone openpi (unmodified, pinned) and set up its server environment with uv.
# The policy server runs from this clone; this repo never patches it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned openpi commit this demo was built against (main as of 2026-09).
OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479

if [ ! -d vendor/openpi/.git ]; then
    mkdir -p vendor
    git clone https://github.com/Physical-Intelligence/openpi.git vendor/openpi
fi
git -C vendor/openpi fetch origin "$OPENPI_COMMIT"
git -C vendor/openpi checkout "$OPENPI_COMMIT"
git -C vendor/openpi submodule update --init --recursive

cd vendor/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo "openpi server environment ready (vendor/openpi @ $OPENPI_COMMIT)"
