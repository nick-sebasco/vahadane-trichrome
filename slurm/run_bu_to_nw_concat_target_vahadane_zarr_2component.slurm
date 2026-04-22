#!/bin/bash
#SBATCH --job-name=bu2nw-zarr-c2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --array=0-28
#SBATCH --output=/home/sebasn/vahadane-trichrome/outputs/slurm_logs/bu2nw-zarr-c2-%A_%a.out
#SBATCH --error=/home/sebasn/vahadane-trichrome/outputs/slurm_logs/bu2nw-zarr-c2-%A_%a.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
SOURCE_ROOT="/projects/activities/mahoney-tb-lung/SkinImages/BUTrichrome/Zarrs"
REFERENCE_PNG="${REPO_DIR}/examples/example_data/NW_concat_targets/NW_ref_good_level4_concat.png"
OUTPUT_ROOT="/flashscratch/sebasn/BU_vahadane"
TARGET_FIT_STATE="${OUTPUT_ROOT}/target_fit_state_level4_concat_2component.npz"
FIT_LOCK_DIR="${OUTPUT_ROOT}/.target_fit_state_lock"

mkdir -p "${OUTPUT_ROOT}" "${REPO_DIR}/outputs/slurm_logs"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-0}"

mapfile -t SOURCE_ZARRS < <(find "${SOURCE_ROOT}" -maxdepth 1 -mindepth 1 -type d -name '*.zarr' -printf '%f\n' | sort)

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${#SOURCE_ZARRS[@]}" ]]; then
  printf 'Array index %s is out of range for %s BU zarrs.\n' "${SLURM_ARRAY_TASK_ID}" "${#SOURCE_ZARRS[@]}" >&2
  exit 1
fi

SOURCE_NAME="${SOURCE_ZARRS[${SLURM_ARRAY_TASK_ID}]}"
SOURCE_ZARR="${SOURCE_ROOT}/${SOURCE_NAME}"
OUTPUT_ZARR="${OUTPUT_ROOT}/${SOURCE_NAME}"

printf 'Normalizing BU zarr %s using target %s\n' "${SOURCE_ZARR}" "${REFERENCE_PNG}"
printf 'Output zarr: %s\n' "${OUTPUT_ZARR}"

if [[ ! -f "${TARGET_FIT_STATE}" ]]; then
  if mkdir "${FIT_LOCK_DIR}" 2>/dev/null; then
    trap 'rmdir "${FIT_LOCK_DIR}" 2>/dev/null || true' EXIT
    printf 'Creating shared target fit state at %s\n' "${TARGET_FIT_STATE}"
    poetry run python -u LOCAL/LOCAL_apply_vahadane_to_zarr.py \
      --reference-png "${REFERENCE_PNG}" \
      --target-fit-state "${TARGET_FIT_STATE}" \
      --fit-target-only \
      --luminosity-threshold 0.65 \
      --use-connected-components true \
      --min-component-size-fraction 5e-4 \
      --min-component-size-relative-to-largest 1e-2 \
      --cumulative-foreground-coverage 0.995 \
      --connected-components-connectivity 2 \
      --connected-components-fail-safe true \
      --regularizer 0.05 \
      --n-components 2 \
      --sort-mode none \
      --random-state 0 \
      --fit-algorithm lars \
      --transform-algorithm lasso_lars \
      --dl-max-iter 100 \
      --dl-transform-max-iter 1000 \
      --dl-n-jobs "${THREADS}" \
      --max-concentration-scale-factor 4.0
    rmdir "${FIT_LOCK_DIR}"
    trap - EXIT
  else
    printf 'Waiting for shared target fit state %s\n' "${TARGET_FIT_STATE}"
    while [[ ! -f "${TARGET_FIT_STATE}" ]]; do
      sleep 10
    done
  fi
fi

poetry run python -u LOCAL/LOCAL_apply_vahadane_to_zarr.py \
  --source-zarr "${SOURCE_ZARR}" \
  --reference-png "${REFERENCE_PNG}" \
  --output-zarr "${OUTPUT_ZARR}" \
  --target-fit-state "${TARGET_FIT_STATE}" \
  --source-fit-level 4 \
  --overwrite \
  --apply-source-tissue-mask \
  --luminosity-threshold 0.65 \
  --use-connected-components true \
  --min-component-size-fraction 5e-4 \
  --min-component-size-relative-to-largest 1e-2 \
  --cumulative-foreground-coverage 0.995 \
  --connected-components-connectivity 2 \
  --connected-components-fail-safe true \
  --regularizer 0.05 \
  --n-components 2 \
  --sort-mode none \
  --random-state 0 \
  --fit-algorithm lars \
  --transform-algorithm lasso_lars \
  --dl-max-iter 100 \
  --dl-transform-max-iter 1000 \
  --dl-n-jobs "${THREADS}" \
  --max-concentration-scale-factor 4.0

printf 'Finished normalization for %s\n' "${SOURCE_NAME}"
