#!/bin/bash
#SBATCH --job-name=puzzle_seg
#SBATCH --output=logs/puzzle_seg_%j.out
#SBATCH --error=logs/puzzle_seg_%j.err
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

python train_model.py