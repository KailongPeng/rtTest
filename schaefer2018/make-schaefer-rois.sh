#!/bin/bash
#SBATCH --partition=short,scavenge,day,long
#SBATCH --job-name=schaeferMasks
#SBATCH --time=40:00
#SBATCH --output=./logs/schaeferMask-%j.out
#SBATCH --mem=2g

set -e #stop immediately encountering error
module load AFNI/2018.08.28 ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh ; module load miniconda ; source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud
sub=$1
mkdir -p ./${sub}


STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz
THR=10


ROIpath=/gpfs/milgram/scratch/turk-browne/tsy6/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI

#register deskulled roi to individual subject t1
WANG2FUNC=./../wang2014/${sub}/wang2func.mat
TEMPLATE="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz"
TEMPLATE_bet=./../wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz
bet ${TEMPLATE} ${TEMPLATE_bet}
WANGINFUNC=./../wang2014/${sub}/wanginfunc.nii.gz
if [ -f "$WANG2FUNC" ]; then
    echo "xfm mat exists"
else 
    echo "xfm mat does not exist"
    # stand_funcOrien=./${sub}/wang_funcOrien.nii.gz
    # echo python -u /gpfs/milgram/project/turk-browne/projects/rtTest/orien_trans.py $STAND $TEMPLATE_bet $stand_funcOrien
    # python -u /gpfs/milgram/project/turk-browne/projects/rtTest/orien_trans.py $STAND $TEMPLATE_bet $stand_funcOrien
    # flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC

    flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC
fi

atlas=Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz

#convert ROIs from wang2014 standard space to individual T1 space
for ROI in {1..300}; do
  INPUT=$ROIpath/$atlas # schaefer2018 standard space
  OUTPUT=./${sub}/${ROI}.nii.gz #individual T1 space ROI outputs
  fslmaths $INPUT -thr $ROI -uthr $ROI -bin $OUTPUT
  flirt -ref $TEMPLATE_bet -in $OUTPUT -out $OUTPUT -applyxfm -init $WANG2FUNC
  echo flirt -ref $TEMPLATE_bet -in $OUTPUT -out $OUTPUT -applyxfm -init $WANG2FUNC
  fslmaths $OUTPUT -thr 0.5 -bin $OUTPUT
done
  

echo done