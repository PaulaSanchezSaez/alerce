#!/bin/bash

#SBATCH --job-name=FEATURES
#SBATCH --partition=general
#SBATCH -n 120
#SBATCH --ntasks-per-node=40
#SBATCH --output=features_%j.out
#SBATCH --error=features_%j.err

export OMP_NUM_THREADS=1
ml Anaconda3/5.3.0
source ~/features/VENV/bin/activate
srun ~/features/VENV/bin/python3.6 ~/features/scripts/non_det_features_mpi.py ~/features/data/20191119/input/splitted ~/features/data/20191119/output/non_det
