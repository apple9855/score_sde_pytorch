#!/usr/bin/env bash
set -euo pipefail

echo "===== [01] BOOTSTRAP START ====="

echo "[1] System / GPU info"
uname -a || true
nvidia-smi || true

echo "[2] Python / pip"
python -V
pip -V

echo "[3] OS build deps (best-effort)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update || true
  apt-get install -y build-essential ninja-build git curl || true
  echo "[3.1] ninja sanity"
  command -v ninja >/dev/null 2>&1 && ninja --version || echo "ninja not found (may be OK if no extensions build)"
else
  echo "apt-get not available, skipping system deps"
fi

echo "[4] Upgrade pip toolchain"
pip install -U pip setuptools wheel

echo "[5] Install Python requirements (torch already present)"
pip install --no-cache-dir -r engineering_validation/requirements-validation.txt

echo "[6] TensorFlow GPU isolation (CPU-only enforcement test)"
# Strong isolation: hide GPU from TF (does not affect torch running in other processes)
CUDA_VISIBLE_DEVICES="" python - <<'PY'
import tensorflow as tf
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("TF sees GPUs:", gpus)
if gpus:
    raise RuntimeError("ERROR: TensorFlow should not see GPU!")
else:
    print("OK: TensorFlow is CPU-only (or GPU hidden)")
PY

echo "[7] Torch CUDA sanity check"
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== [01] DONE ====="