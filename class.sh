#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --job-name class
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=10000
#SBATCH --output=logs/class_%A_%a.out
#SBATCH --requeue
#SBATCH --partition=short,day,scavenge,long,verylong
#SBATCH --mail-type FAIL
#SBATCH --mail-user=kp578

# Set up the environment
# module load FSL/5.0.9
# module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
# module load nilearn
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud

echo python -u class.py $1 $SLURM_ARRAY_TASK_ID
python -u class.py $1 $SLURM_ARRAY_TASK_ID