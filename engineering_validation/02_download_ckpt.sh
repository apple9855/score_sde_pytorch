#!/usr/bin/env bash
set -euo pipefail

echo "===== [02] DOWNLOAD CHECKPOINTS + STATS ====="

# -----------------------------
# Root exp directory (Song repo convention)
# -----------------------------
EXP_DIR="exp"
mkdir -p "${EXP_DIR}"

# -----------------------------
# VE-SDE: CIFAR-10 NCSN++
# -----------------------------
VE_DIR="${EXP_DIR}/ve/cifar10_ncsnpp_continuous"
VE_CKPT="${VE_DIR}/checkpoints/checkpoint_24.pth"
VE_GDRIVE_ID="1jFmheW6vFKUzvPCW2uCgaUskrGcomj8u"

# -----------------------------
# VP-SDE: CIFAR-10 DDPM++
# -----------------------------
VP_DIR="${EXP_DIR}/vp/cifar10_ddpmpp_continuous"
VP_CKPT="${VP_DIR}/checkpoints/checkpoint_26.pth"
VP_GDRIVE_ID="1A9u9CrvUbxho6j3TjxVOtc-K9n701seT"

# IMPORTANT: ensure checkpoints subdirs exist (Mode B)
mkdir -p "${VE_DIR}/checkpoints"
mkdir -p "${VP_DIR}/checkpoints"

# -----------------------------
# CIFAR-10 stats (FID / KID / IS)
# -----------------------------
STATS_DIR="assets/stats"
STATS_PATH="${STATS_DIR}/cifar10_stats.npz"
STATS_GDRIVE_ID="14UB27-Spi8VjZYKST3ZcT8YVhAluiFWI"

mkdir -p "${STATS_DIR}"

# -----------------------------
# Download helper
# -----------------------------
download_ckpt () {
  local FILE_ID="$1"
  local OUT_PATH="$2"

  if [[ -f "${OUT_PATH}" ]]; then
    echo "[SKIP] Exists: ${OUT_PATH}"
  else
    echo "[DOWNLOAD] ${OUT_PATH}"
    gdown --id "${FILE_ID}" -O "${OUT_PATH}"
  fi

  echo "[CHECKSUM]"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${OUT_PATH}"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${OUT_PATH}"
  else
    echo "<no sha256 tool available>"
  fi
  echo

  echo "[SIZE]"
  ls -lh "${OUT_PATH}" || true
  echo
}

# -----------------------------
# Execute downloads
# -----------------------------

echo "[VE-SDE] NCSN++ CIFAR-10"
download_ckpt "${VE_GDRIVE_ID}" "${VE_CKPT}"

echo "[VP-SDE] DDPM++ CIFAR-10"
download_ckpt "${VP_GDRIVE_ID}" "${VP_CKPT}"

echo "[stats] CIFAR-10"
download_ckpt "${STATS_GDRIVE_ID}" "${STATS_PATH}"


echo "===== [02] DONE ====="