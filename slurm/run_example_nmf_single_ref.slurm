#!/bin/bash
#SBATCH --job-name=vah-nmf-example
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/flashscratch/sebasn/vahadane_runs/example_nmf_single_ref/slurm-%j.out
#SBATCH --error=/flashscratch/sebasn/vahadane_runs/example_nmf_single_ref/slurm-%j.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
SOURCE_IMAGE="${REPO_DIR}/examples/example_data/BU_2_-_2024-08-23_08.38.00.png"
REFERENCE_IMAGE="${REPO_DIR}/examples/example_data/NW_NW_Trichrome_Box_1_LF_RF_SSCMH27_BASE_LF_04-27-11_-_2022-11-10_16.02.34_left.png"
OUTPUT_DIR="/flashscratch/sebasn/vahadane_runs/example_nmf_single_ref"
OUTPUT_IMAGE="${OUTPUT_DIR}/normalized.png"
FIT_STATE_PATH="${OUTPUT_DIR}/fit_state.npz"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

poetry run vahadane-trichrome \
  --source "${SOURCE_IMAGE}" \
  --reference "${REFERENCE_IMAGE}" \
  --output "${OUTPUT_IMAGE}" \
  --artifact-dir "${OUTPUT_DIR}" \
  --save-swatches \
  --save-roi-images \
  --save-fit-state "${FIT_STATE_PATH}" \
  --apply-source-tissue-mask \
  --backend nmf \
  --luminosity-threshold 0.65 \
  --use-connected-components true \
  --min-component-size-fraction 5e-4 \
  --min-component-size-relative-to-largest 1e-2 \
  --cumulative-foreground-coverage 0.995 \
  --connected-components-connectivity 2 \
  --connected-components-fail-safe true \
  --regularizer 1e-4 \
  --n-components 3 \
  --sort-mode none \
  --max-tissue-pixels 8000 \
  --random-state 0 \
  --nmf-init nndsvdar \
  --nmf-solver cd \
  --nmf-beta-loss frobenius \
  --nmf-tol 1e-4 \
  --nmf-max-iter 3000 \
  --nmf-shuffle false \
  --max-concentration-scale-factor 4.0

printf 'NMF normalization complete. Output: %s\n' "${OUTPUT_IMAGE}"