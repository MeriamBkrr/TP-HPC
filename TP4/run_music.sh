#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J music-lstm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs TP4/generated

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

PYTHONPATH=TP4 python -u TP4/train_lstm_large.py
