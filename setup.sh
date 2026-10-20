#!/usr/bin/env bash
# Prepare a standalone TelcoBench checkout after cloning.
#
# This script creates only repository-local state: .venv and, if absent, .env.
# It never downloads model weights, writes credentials, or requires sudo.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required but was not found on PATH." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found; installing it into the current user's Python environment..."
  python3 -m pip install --user --upgrade uv
  user_bin=$(python3 -m site --user-base)/bin
  export PATH="$user_bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation completed but is not on PATH. Add your Python user bin directory to PATH and rerun." >&2
  exit 1
fi

echo "Creating/synchronizing the package-local environment..."
eval_python="${NETWORK_LLM_EVAL_PYTHON:-python3.11}"
if ! command -v "$eval_python" >/dev/null 2>&1; then
  echo "Required Python interpreter is unavailable: $eval_python" >&2
  echo "Install CPython 3.11 or set NETWORK_LLM_EVAL_PYTHON to a supported 3.10–3.12 interpreter." >&2
  exit 1
fi
# Recreate only the disposable repository-local virtual environment. This
# matches the CPython 3.11 judge transport validated in MOPD.
uv venv --clear --python "$eval_python" .venv
uv sync --extra huggingface --extra tui --extra judge --extra dev

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in NETWORK_LLM_EVAL_RELEASE_ROOT and judge settings before a model run."
else
  echo "Keeping existing .env."
fi

mkdir -p outputs

echo "Validating the bundled held-out manifest..."
.venv/bin/telcobench validate datasets/MANIFEST.json

cat <<'EOF'

Setup complete.

Next steps:
  1. Edit .env with your local approved artifact-bundle root and judge identity.
  2. Validate a configured target without loading weights:
       .venv/bin/telcobench run-config --config config/default.yaml --target cpt_base --dry-run
  3. Start the read-only monitor:
       .venv/bin/telcobench tui --config config/default.yaml --refresh-seconds 2
EOF
