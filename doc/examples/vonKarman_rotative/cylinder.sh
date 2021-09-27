#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --begin=now
#SBATCH --job-name=cylinder
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antonio.gimenez-nadal@student.isae-supaero.fr
#SBATCH --output=./results/slurm.%j.out
#SBATCH --error=./results/slurm.%j.err



source $HOME/fluidnet_env/bin/activate

simulate
