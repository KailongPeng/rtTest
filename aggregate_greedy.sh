#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/agg-%j.out
#SBATCH --job-name aggregate_greedy
#SBATCH --partition=verylong,long,day,short
#SBATCH --time=20:00:00
#SBATCH --mem=10000
#SBATCH -n 5
#SBATCH --mail-type FAIL
#SBATCH --mail-user=kp578

# Set up the environment
# module load FSL/5.0.9
# module load Python/Anaconda3
# module load FreeSurfer/6.0.0
# module load BXH_XCEDE_TOOLS
# module load brainiak
# module load nilearn

subject=$1
dataloc=$2
roiloc=$3
Nregions=$4

# Run the python scripts
echo "running searchlight"
echo source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
python -u ./aggregate_greedy.py $subject $dataloc $roiloc $Nregions
# python -u ./aggregate.py $subject $dataloc $roiloc $Nregions
