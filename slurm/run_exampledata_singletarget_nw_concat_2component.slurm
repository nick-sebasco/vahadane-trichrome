#!/bin/bash
#SBATCH --job-name=vah-exconcat-c2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=/flashscratch/sebasn/vahadane_runs/example_nw_concat_2component/slurm-%j.out
#SBATCH --error=/flashscratch/sebasn/vahadane_runs/example_nw_concat_2component/slurm-%j.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
TARGET_IMAGE="${REPO_DIR}/examples/example_data/NW_2_CONCAT.png"
BU_SOURCE_IMAGE="${REPO_DIR}/examples/example_data/BU_6_-_2024-08-23_08.50.20.png"
KD_SOURCE_IMAGE="${REPO_DIR}/examples/example_data/KD_V_tri_center.png"
OUTPUT_ROOT="/flashscratch/sebasn/vahadane_runs/example_nw_concat_2component"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

run_single_target_job() {
  local run_name="$1"
  local source_image="$2"
  local output_dir="${OUTPUT_ROOT}/${run_name}"
  local output_image="${output_dir}/normalized.png"
  local fit_state_path="${output_dir}/fit_state.npz"

  mkdir -p "${output_dir}"

  printf 'Starting run %s\n' "${run_name}"
  printf '  source: %s\n' "${source_image}"
  printf '  target: %s\n' "${TARGET_IMAGE}"
  printf '  output: %s\n' "${output_dir}"

  poetry run vahadane-trichrome \
    --source "${source_image}" \
    --reference "${TARGET_IMAGE}" \
    --output "${output_image}" \
    --artifact-dir "${output_dir}" \
    --save-swatches \
    --save-roi-images \
    --save-fit-state "${fit_state_path}" \
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

  printf 'Completed run %s. Output: %s\n' "${run_name}" "${output_image}"
}

run_single_target_job "bu6_to_nw_concat" "${BU_SOURCE_IMAGE}"
run_single_target_job "kd_center_to_nw_concat" "${KD_SOURCE_IMAGE}"

printf 'All 2-component concatenated-target runs completed. Root: %s\n' "${OUTPUT_ROOT}"
