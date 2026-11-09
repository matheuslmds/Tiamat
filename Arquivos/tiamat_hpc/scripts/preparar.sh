#!/bin/bash
#SBATCH --job-name=tiamat_divisoes_mlms
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/preparar_%j.out
#SBATCH --error=logs/preparar_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tiamat_MLMS
cd /home/matheus2610033/Ilum/sem_2/tiamat/tiamat_hpc

python -u scripts/preparar_divisoes.py

