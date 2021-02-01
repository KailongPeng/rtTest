#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/maskmaker-%j.out
#SBATCH --job-name searchlight
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --mem=10000
#SBATCH -n 1

# Set up the environment
module load FSL/5.0.9
module load Python/Anaconda3
module load FreeSurfer/6.0.0
module load BXH_XCEDE_TOOLS
module load brainiak
module load nilearn

subject=$1

# Run the python scripts
echo "running searchlight"

python -u ./assemble.py $subject
