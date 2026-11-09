#!/bin/bash
#SBATCH --job-name=tiamat_MLMS
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/tiamat_%A_%a.out
#SBATCH --error=logs/tiamat_%A_%a.err

set -euo pipefail

# Ativar o ambiente Python
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tiamat_MLMS

cd /home/matheus2610033/Ilum/sem_2/tiamat/tiamat_hpc

# Evitar paralelismo excessivo
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Identificar o algoritmo
MODELOS=(
    "random_forest"
    "extra_trees"
    "hist_gradient_boosting"
    "xgboost"
)

MODELO=${MODELOS[$SLURM_ARRAY_TASK_ID]}

echo "Algoritmo: $MODELO"
echo "Nó: $(hostname)"
echo "CPUs alocadas: $SLURM_CPUS_PER_TASK"
echo "Início: $(date)"

python -u scripts/otimizar.py \
    --modelo "$MODELO" \
    --trials 700 \

echo "Fim: $(date)"
