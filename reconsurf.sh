#!/usr/bin/env bash
# Input python command to be submitted as a job
#SBATCH --output=logs/FS-%j.out
#SBATCH --job-name FS
#SBATCH --partition=verylong,day
#SBATCH --time=24:00:00
#SBATCH --mem=10000

module load FSL/5.0.9
module load FreeSurfer/6.0.0
#module load BXH_XCEDE_TOOLS
#module load nilearn
module load AFNI

subject=$1
source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
rttPath=/gpfs/milgram/project/turk-browne/projects/rtTest/
subjectFolder=/gpfs/milgram/project/turk-browne/projects/rtTest/subjects/${subject}/freesurfer/
rm -r ${subjectFolder}
mkdir -p ${subjectFolder}

# process steps 1-5 
recon-all -i ${rttPath}/subjects/${subject}/anat/anat_reorien.nii.gz -autorecon1 -notal-check -subjid ${subject} -sd ${subjectFolder};

# process steps 6-23
recon-all -autorecon2 -subjid ${subject} -sd ${subjectFolder};

# process stages 24-31
recon-all -autorecon3 -subjid ${subject} -sd ${subjectFolder};
