#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=volta
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=10:00:00
#SBATCH --mem=16000
#SBATCH --begin=now
#SBATCH --job-name=test_sim
#SBATCH --output=./results/slurm.%j.out
#SBATCH --error=./results/slurm.%j.err


source /tmpdir/ajuriail/neuralsim/neural_sim_env/bin/activate

simulate
