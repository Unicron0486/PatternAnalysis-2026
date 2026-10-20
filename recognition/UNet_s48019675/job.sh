#!/bin/bash
#SBATCH --job-name=H2dUnetH
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch
# Run the script
python predict.py