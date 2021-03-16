#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/SUMA_Make_Spec_FS-%j.out
#SBATCH --job-name SUMA_Make_Spec_FS
#SBATCH --partition=short,scavenge,day
#SBATCH --time=2:00:00
#SBATCH --mem=10000
module load AFNI
module load FreeSurfer/6.0.0
echo $FREESURFER_HOME # /gpfs/milgram/apps/hpc.rhel7/software/FreeSurfer/6.0.0
subject=$1
subjectFolder=/gpfs/milgram/project/turk-browne/projects/rtTest/subjects/${subject}/freesurfer/${subject}/
cd ${subjectFolder}
@SUMA_Make_Spec_FS -sid ${subject}
