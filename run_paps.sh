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
srun ~/features/VENV/bin/python3.6 ~/features/scripts/compute_features_paps_mpi.py /home/astrouser/features/data/20191017/input/splitted  /home/astrouser/features/data/20191017/output/paps


