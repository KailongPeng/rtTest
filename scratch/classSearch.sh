#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=maskmaker-%j.out
#SBATCH --job-name searchlight
#SBATCH --partition=verylong
#SBATCH --time=20:00:00
#SBATCH --mem=100000
#SBATCH -n 25

# Set up the environment
module load FSL/5.0.9
module load Python/Anaconda3
module load FreeSurfer/6.0.0
module load BXH_XCEDE_TOOLS
module load brainiak
module load nilearn

echo $1

# Run the python scripts
#python -u ./makeMasks.py 0110171 neurosketch
echo "mask made"
echo "running searchlight"
mpirun -n 25 python -u ./classSearch.py 0110171 $1 neurosketch $2
