#!/usr/bin/env bash
# Set up the LIBERO simulator client environment (separate from the server
# env, exactly as openpi's examples/libero/README.md documents).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d vendor/openpi/.git ]; then
    echo "run scripts/setup_openpi.sh first" >&2
    exit 1
fi

uv venv --python 3.8 .venv-client
# shellcheck disable=SC1091
source .venv-client/bin/activate
uv pip sync vendor/openpi/examples/libero/requirements.txt \
    vendor/openpi/third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e vendor/openpi/packages/openpi-client
uv pip install -e vendor/openpi/third_party/libero

# LIBERO asks interactively about a dataset path on first import; answer "no"
# so it writes its default ~/.libero/config.yaml (we never download datasets).
echo n | PYTHONPATH=vendor/openpi/third_party/libero .venv-client/bin/python \
    -c "import libero.libero" >/dev/null
echo "client environment ready (.venv-client)"
