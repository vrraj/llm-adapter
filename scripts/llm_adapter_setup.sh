#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# llm-adapter — Quick local setup (macOS/Linux)
# -----------------------------------------------------------------------------
# What this script does:
#   1) Validates basic prerequisites (python3, make)
#   2) Creates a local Python virtualenv (.venv)
#   3) Installs llm-adapter in editable mode (pip install -e .)
#   4) Ensures a .env file exists (copies from .env.example if present)
#   5) Prints next steps for running the FastAPI demo (via Makefile)
#
# How to run (from repo root):
#   bash scripts/llm_adapter_setup.sh
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

need python3
need make

cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"

echo
echo "[1/3] Preparing .env ..."
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "  - Created .env from .env.example"
  else
    : > .env
    echo "  - Created empty .env (fill in any required keys manually)"
  fi
else
  echo "  - .env already exists (leaving as-is)"
fi

echo
echo "[2/3] Creating Python virtualenv in .venv (if needed) ..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
  echo "  - Created .venv/"
else
  echo "  - .venv/ already exists (reusing)"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo
echo "[3/3] Installing llm-adapter in editable mode ..."
python -m pip install --upgrade pip
pip install -e .

deactivate

echo
echo "✅ Setup complete for llm-adapter. Next steps:"
echo "  1) Activate your venv:"
echo "       source .venv/bin/activate"
echo "  2) Start the FastAPI demo (foreground):"
echo "       make start"
echo "     or background with logs:"
echo "       make start-bg"
echo "  3) Open the demo UI:"
echo "       http://localhost:8100/ui/"
