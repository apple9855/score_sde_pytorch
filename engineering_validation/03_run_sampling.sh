#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------
# Force working directory to repo root
# engineering_validation/03_run_sampling.sh  -> repo root is one level up
# ----------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "===== [03] RUN SAMPLING (SHOWCASE-GRADE MINI EVAL) ====="
echo "[PWD] $(pwd)"
echo "[SYS ] $(uname -a || true)"
echo "[PY  ] $(python -V 2>&1)"
echo "[PIP ] $(pip -V 2>&1)"

# ---- Required tools sanity (best-effort) ----
command -v tee  >/dev/null 2>&1 || { echo "ERROR: tee not found"; exit 1; }
command -v find >/dev/null 2>&1 || { echo "ERROR: find not found"; exit 1; }

# ---- Check required assets ----
test -f "assets/stats/cifar10_stats.npz" || {
  echo "ERROR: assets/stats/cifar10_stats.npz not found (expected in repo)."
  exit 1
}

# ---- (Mode B) Workdirs: checkpoints must live under workdir/checkpoints/ ----
VE_WORKDIR="exp/ve/cifar10_ncsnpp_continuous"
VP_WORKDIR="exp/vp/cifar10_ddpmpp_continuous"

test -f "${VE_WORKDIR}/checkpoints/checkpoint_24.pth" || {
  echo "ERROR: Missing ${VE_WORKDIR}/checkpoints/checkpoint_24.pth"
  exit 1
}
test -f "${VP_WORKDIR}/checkpoints/checkpoint_26.pth" || {
  echo "ERROR: Missing ${VP_WORKDIR}/checkpoints/checkpoint_26.pth"
  exit 1
}

# ---- Configs (update if your paths differ) ----
VE_CFG="configs/ve/cifar10_ncsnpp_continuous.py"
VP_CFG="configs/vp/cifar10_ddpmpp_continuous.py"
test -f "${VE_CFG}" || { echo "ERROR: Missing config ${VE_CFG}"; exit 1; }
test -f "${VP_CFG}" || { echo "ERROR: Missing config ${VP_CFG}"; exit 1; }

# ---- Logging & artifacts ----
LOG_DIR="engineering_validation/logs"
ART_DIR="engineering_validation/artifacts"
mkdir -p "${LOG_DIR}" "${ART_DIR}"

timestamp() { date +"%Y%m%d_%H%M%S"; }

gpu_snapshot () {
  echo "----- [GPU SNAPSHOT] -----"
  nvidia-smi || echo "nvidia-smi not available"
  echo "--------------------------"
}

# Copy key artifacts if present (best-effort, non-invasive)
collect_artifacts () {
  local NAME="$1"
  local WORKDIR="$2"
  local DEST="${ART_DIR}/${NAME}"
  mkdir -p "${DEST}"

  echo "[COLLECT] Searching report_*.npz under: ${WORKDIR}"
  local REPORTS
  REPORTS="$(find "${WORKDIR}" -maxdepth 6 -type f -name "report_*.npz" 2>/dev/null || true)"

  if [[ -z "${REPORTS}" ]]; then
    echo "[COLLECT] No report_*.npz found. (Maybe eval outputs are stored elsewhere.)"
  else
    echo "[COLLECT] Found reports:"
    echo "${REPORTS}"
    while IFS= read -r f; do
      [[ -z "${f}" ]] && continue
      cp -n "${f}" "${DEST}/" || true
    done <<< "${REPORTS}"
  fi

  echo "[COLLECT] Searching samples/statistics under: ${WORKDIR}"
  local FILES
  FILES="$(find "${WORKDIR}" -maxdepth 6 -type f \( -name "samples_*.npz" -o -name "statistics_*.npz" \) 2>/dev/null || true)"
  if [[ -n "${FILES}" ]]; then
    while IFS= read -r f; do
      [[ -z "${f}" ]] && continue
      # Preserve structure when possible; fallback to flat copy
      cp -n --parents "${f}" "${DEST}/" 2>/dev/null || cp -n "${f}" "${DEST}/" || true
    done <<< "${FILES}"
  fi

  echo "[COLLECT] Artifact files under ${DEST}:"
  (cd "${DEST}" && find . -type f | sort) || true
}

run_eval () {
  local NAME="$1"
  local CFG="$2"
  local WORKDIR="$3"
  local TS
  TS="$(timestamp)"
  local LOGFILE="${LOG_DIR}/03_${NAME}_${TS}.log"

  echo
  echo "===== RUN: ${NAME} ====="
  echo "[CONFIG ] ${CFG}"
  echo "[WORKDIR] ${WORKDIR}"
  echo "[LOG    ] ${LOGFILE}"
  echo "[START ] ${TS}"
  echo

  gpu_snapshot | tee -a "${LOGFILE}"

  local START_SEC
  START_SEC="$(date +%s)"

  # Run + log capture (stdout+stderr)
  python -u main.py \
    --config="${CFG}" \
    --workdir="${WORKDIR}" \
    --mode=eval 2>&1 | tee "${LOGFILE}"

  local END_SEC
  END_SEC="$(date +%s)"
  local ELAPSED=$((END_SEC - START_SEC))

  echo
  echo "[TIME] ${NAME} elapsed: ${ELAPSED} seconds" | tee -a "${LOGFILE}"
  gpu_snapshot | tee -a "${LOGFILE}"

  # Collect showcase artifacts (best-effort)
  collect_artifacts "${NAME}" "${WORKDIR}" | tee -a "${LOGFILE}"

  echo "===== RUN DONE: ${NAME} =====" | tee -a "${LOGFILE}"
}

# ----------------------------------
# Execute (VE then VP)
# ----------------------------------
run_eval "VE_ncsnpp_ckpt24" "${VE_CFG}" "${VE_WORKDIR}"
run_eval "VP_ddpmpp_ckpt26" "${VP_CFG}" "${VP_WORKDIR}"

echo
echo "===== [03] DONE ====="
echo "[LOGS ] ${LOG_DIR}"
echo "[ART  ] ${ART_DIR}"