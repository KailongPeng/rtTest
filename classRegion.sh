#!/usr/bin/env bash
#SBATCH --job-name searchlight
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=100000
#SBATCH --output=logs/searchlight_%A_%a.out
#SBATCH --requeue
#SBATCH --partition=short,day,scavenge,long,verylong
#SBATCH --mail-type FAIL
#SBATCH --mail-user=kp578


# Set up the environment
module load FSL/5.0.9
module load Python/Anaconda3
module load FreeSurfer/6.0.0
module load BXH_XCEDE_TOOLS
module load brainiak
module load nilearn

subject=$1
dataloc=$2
roiloc=$3
# roinum=$4
roinum=$SLURM_ARRAY_TASK_ID
# roihemi=$5
roihemi=$4 # this can be blank for Schaefer atlas

# Run the python scripts
echo "running searchlight"
mkdir -p /gpfs/milgram/project/turk-browne/projects/rtTest/$roiloc/$subject/output

python -u ./classRegion.py $subject $dataloc $roiloc $roinum $roihemi
