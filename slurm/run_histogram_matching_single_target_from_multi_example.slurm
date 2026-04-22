#!/bin/bash
#SBATCH --job-name=hm-single-vs-multi
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/sebasn/vahadane-trichrome/outputs/histogram_matching_single_target_from_multi_example/slurm-%j.out
#SBATCH --error=/home/sebasn/vahadane-trichrome/outputs/histogram_matching_single_target_from_multi_example/slurm-%j.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
OUTPUT_ROOT="${REPO_DIR}/outputs/histogram_matching_single_target_from_multi_example"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
export MPLCONFIGDIR="${OUTPUT_ROOT}/.matplotlib"

poetry run python examples/histogram_matching_single_target_from_multi_example.py

printf 'Histogram matching single-target comparison complete. Root: %s\n' "${OUTPUT_ROOT}"
