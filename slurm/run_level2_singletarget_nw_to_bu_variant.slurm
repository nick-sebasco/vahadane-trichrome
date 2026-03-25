#!/bin/bash
#SBATCH --job-name=vah-single-l2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=/home/sebasn/vahadane-trichrome/outputs/slurm_logs/vah-single-l2-%j.out
#SBATCH --error=/home/sebasn/vahadane-trichrome/outputs/slurm_logs/vah-single-l2-%j.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
TEST_IMAGE_DIR="/flashscratch/sebasn/vahadane_test_images"
OUTPUT_DIR="/flashscratch/sebasn/vahadane_runs/bu10_to_nw_single_level2_cd_omp"
SOURCE_PNG="${TEST_IMAGE_DIR}/BU_level2_10_-_2024-08-23_09.03.00.png"
REFERENCE_PNG="${TEST_IMAGE_DIR}/NW_level2_NW_Trichrome_Boc_1_LF_LA_Controls_AAA034979_0001_-_2022-11-11_15.09.03.png"
OUTPUT_IMAGE="${OUTPUT_DIR}/normalized.png"
FIT_STATE_PATH="${OUTPUT_DIR}/fit_state.npz"

SOURCE_ZARR="/projects/activities/mahoney-tb-lung/SkinImages/BUTrichrome/Zarrs/10_-_2024-08-23_09.03.00.zarr"
REFERENCE_ZARR="/projects/activities/mahoney-tb-lung/SkinImages/Zarrs/NW_Trichrome_Boc_1_LF_LA_Controls_AAA034979_0001_-_2022-11-11_15.09.03.zarr"

mkdir -p "${OUTPUT_DIR}" "${TEST_IMAGE_DIR}"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

ensure_level2_png() {
  local zarr_path="$1"
  local png_path="$2"

  if [[ -f "${png_path}" ]]; then
    printf 'Using existing PNG: %s\n' "${png_path}"
    return 0
  fi

  printf 'Creating PNG from %s -> %s\n' "${zarr_path}" "${png_path}"
  poetry run python - <<PY
import os
import matplotlib.pyplot as plt
import numpy as np
import zarr

zarr_path = ${zarr_path@Q}
png_path = ${png_path@Q}

arr = np.asarray(zarr.open(zarr_path, mode="r")["0"]["2"])
image = np.squeeze(arr)
if image.ndim != 3:
    raise ValueError(f"Expected a 3D image after squeeze, got {image.shape}.")
if image.shape[0] in (3, 4):
    image = np.moveaxis(image[:3, ...], 0, -1)
elif image.shape[-1] in (3, 4):
    image = image[..., :3]
else:
    raise ValueError(f"Unsupported image shape: {image.shape}")
image = image.astype(np.uint8, copy=False)
os.makedirs(os.path.dirname(png_path), exist_ok=True)
plt.imsave(png_path, image)
print(f"Saved {png_path} with shape {image.shape}")
PY
}

ensure_level2_png "${SOURCE_ZARR}" "${SOURCE_PNG}"
ensure_level2_png "${REFERENCE_ZARR}" "${REFERENCE_PNG}"

poetry run vahadane-trichrome \
  --source "${SOURCE_PNG}" \
  --reference "${REFERENCE_PNG}" \
  --output "${OUTPUT_IMAGE}" \
  --artifact-dir "${OUTPUT_DIR}" \
  --save-swatches \
  --save-roi-images \
  --save-fit-state "${FIT_STATE_PATH}" \
  --apply-source-tissue-mask \
  --luminosity-threshold 0.62 \
  --use-connected-components true \
  --min-component-size-fraction 5e-4 \
  --min-component-size-relative-to-largest 1e-2 \
  --cumulative-foreground-coverage 0.995 \
  --connected-components-connectivity 2 \
  --connected-components-fail-safe true \
  --regularizer 0.08 \
  --n-components 3 \
  --sort-mode none \
  --random-state 0 \
  --fit-algorithm lars \
  --transform-algorithm lasso_cd \
  --dl-max-iter 150 \
  --dl-transform-max-iter 1000 \
  --dl-n-jobs "${THREADS}" \
  --max-concentration-scale-factor 6.0

printf 'Normalization complete. Output: %s\n' "${OUTPUT_IMAGE}"