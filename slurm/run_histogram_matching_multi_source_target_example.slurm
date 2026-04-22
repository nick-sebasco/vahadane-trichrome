#!/bin/bash
#SBATCH --job-name=hm-msmt-example
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/flashscratch/sebasn/vahadane_runs/histogram_matching_multi_source_target_example/slurm-%j.out
#SBATCH --error=/flashscratch/sebasn/vahadane_runs/histogram_matching_multi_source_target_example/slurm-%j.err

set -euo pipefail

REPO_DIR="/home/sebasn/vahadane-trichrome"
OUTPUT_ROOT="/flashscratch/sebasn/vahadane_runs/histogram_matching_multi_source_target_example"

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"

poetry run python examples/histogram_matching_multi_source_target_example.py

printf 'Histogram matching multi-source/target example complete. Root: %s\n' "${OUTPUT_ROOT}"
