#!/usr/bin/env bash
# setup_diffpepdock_env.sh
# Installs DiffPepDock Python dependencies into the diffpepdock venv.
# Run AFTER PyTorch is already installed.
# Usage: bash scripts/setup_diffpepdock_env.sh

set -euo pipefail
VENV="/home/igem/miniconda3/envs/diffpepdock"
PIP="$VENV/bin/pip"

echo "=== Installing DiffPepDock Python dependencies ==="

$PIP install -q \
  "fair-esm==2.0.0" \
  "biotite==0.38.0" \
  "mdtraj==1.9.9" \
  "dm-tree==0.1.8" \
  "easydict==1.11" \
  "hydra-core==1.3.2" \
  "hydra-joblib-launcher==1.2.0" \
  "omegaconf==2.3.0" \
  "ml-collections==0.1.1" \
  "pyrootutils==1.0.4" \
  "python-dotenv==1.0.0" \
  "gputil==1.4.0" \
  "wandb==0.15.12" \
  "tmtools" \
  "openmm>=8.1" \
  "gitpython==3.1.40" \
  "protobuf==4.24.4" \
  "biopython==1.81" \
  "scipy==1.11.4" \
  "scikit-learn==1.3.2" \
  "pandas==2.1.4" \
  "numpy==1.26.4" \
  "tqdm" \
  "matplotlib==3.7.5"

echo ""
echo "=== Installing DiffPepDock package ==="
cd /home/igem/unknown_software/third_party/DiffPepDock
$PIP install -q -e . 2>/dev/null || echo "(no setup.py install needed)"

echo ""
echo "=== Verifying PyTorch + CUDA ==="
$VENV/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== DiffPepDock env setup complete ==="
echo "Python: $VENV/bin/python"
