#!/bin/bash
#SBATCH --job-name=test_imagenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=test_imagenet_%j.out
#SBATCH --error=test_imagenet_%j.err

set -e

echo "=== Informations sur le job ==="
echo "Nœud : $(hostname)"
echo "Date : $(date)"

# Activation de l'environnement Python
source /mnt/hdd/homes/mboukraa/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "=== Vérification du GPU ==="
nvidia-smi

echo "=== Vérification de PyTorch ==="
python -c "import torch; print('PyTorch :', torch.__version__); print('CUDA disponible :', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun GPU détecté')"

echo "=== Test du script ImageNet ==="
python train_imagenet.py

echo "=== Fin du test ==="
