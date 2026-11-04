#!/usr/bin/env bash
#SBATCH --job-name=ion-diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-ion-diffusion-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source .venv-diffusion/bin/activate
: "${MANIFEST:?Set MANIFEST to the data manifest}"
: "${CONFIG:?Set CONFIG to the training YAML}"
: "${OUTPUT:?Set OUTPUT to a new run directory}"
python -m ionrecog.diffusion train --manifest "$MANIFEST" --config "$CONFIG" --output "$OUTPUT" --device cuda
