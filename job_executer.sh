#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu_a100
##SBATCH --partition=gpu_h100
#SBATCH --time=0-01:00:00
#SBATCH --mem=120GB
#SBATCH --output=script_logging/slurm_%A.out
##SBATCH --mail-type=END,FAIL                    # uncomment to be e-mailed when a job ends or fails
##SBATCH --mail-user=<your e-mail address>



# Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0



# srun python defenses/strip.py --attack badnet --model resnet18 --dataset cifar10 --poison_rate 0.05