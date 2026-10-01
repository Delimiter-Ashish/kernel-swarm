#!/bin/bash -l
# One-shot setup on BU SCC. Run from inside a GPU session:  bash setup.sh
set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

module load python3/3.10.12 2>/dev/null || echo "(module load skipped - using current python3)"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"   # keep pip cache out of the small home dir

if [ ! -d kernel-swarm-venv ]; then
  echo ">> creating kernel-swarm-venv in $PROJECT_DIR"
  python3 -m venv --prompt kernel-swarm kernel-swarm-venv
fi
source kernel-swarm-venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

python - <<'PY'
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p results logs
if [ ! -d .git ]; then
  git init -q && git symbolic-ref HEAD refs/heads/main
  echo ">> git repo initialised on branch main"
fi
echo ""
echo "Setup done. Next time just run:  source kernel-swarm-venv/bin/activate"
