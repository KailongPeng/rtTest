#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/maskmaker-%j.out
#SBATCH --job-name searchlight
#SBATCH --partition=long,verylong,short,day,scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=100000
#SBATCH -n 25
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
roinum=$4
roihemi=$5

# Run the python scripts
echo "running searchlight"
mkdir -p /gpfs/milgram/project/turk-browne/projects/rtTest/$roiloc/$subject/output

python -u ./classRegion.py $subject $dataloc $roiloc $roinum $roihemi
