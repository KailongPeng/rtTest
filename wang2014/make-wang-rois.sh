#!/bin/bash
#SBATCH --partition=short,scavenge,day,long
#SBATCH --job-name=wangMasks
#SBATCH --time=40:00
#SBATCH --output=logs/wangMask-%j.out
#SBATCH --mem=2g

set -e #stop immediately encountering error

module load AFNI/2018.08.28 ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.0-centos7_64/etc/fslconf/fsl.sh ; module load miniconda ; source activate /gpfs/milgram/project/turk-browne/users/kp578/CONDA/rtcloud

sub=$1
# sub=1121161 #$1
mkdir -p ./${sub} # save the output files in the current folder


STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz
THR=10
ROIS="roi1 roi2 roi3 roi4 roi5 roi6 roi7 roi8 roi9 roi10 roi11 roi12 roi13 roi14 roi15 roi16 roi17 roi18 roi19 roi20 roi21 roi22 roi23 roi24 roi25"

ROIpath=/gpfs/milgram/project/turk-browne/shared_resources/atlases/ProbAtlas_v4/subj_vol_all

#register deskulled roi to individual subject functional data
WANG2FUNC=./${sub}/wang2func.mat
TEMPLATE="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz"
TEMPLATE_bet=./../wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz
bet ${TEMPLATE} ${TEMPLATE_bet}
WANGINFUNC=./${sub}/wanginfunc.nii.gz


# stand_funcOrien=./${sub}/wang_funcOrien.nii.gz
# echo python -u /gpfs/milgram/project/turk-browne/projects/rtTest/orien_trans.py $STAND $TEMPLATE_bet $stand_funcOrien
# python -u /gpfs/milgram/project/turk-browne/projects/rtTest/orien_trans.py $STAND $TEMPLATE_bet $stand_funcOrien
# flirt -ref $TEMPLATE_bet -in $stand_funcOrien -omat $WANG2FUNC -out $WANGINFUNC
flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC

#convert ROIs from wang2014 standard space to individual T1 space
for ROI in $ROIS; do
  for HEMI in lh rh; do
    INPUT=$ROIpath/perc_VTPM_vol_${ROI}_${HEMI}.nii.gz # wang2014 standard space
    OUTPUT=./${sub}/${ROI}_${HEMI}.nii.gz #individual T1 space ROI outputs
    echo flirt -ref $TEMPLATE_bet -in $INPUT -out $OUTPUT -applyxfm -init $WANG2FUNC
    flirt -ref $TEMPLATE_bet -in $INPUT -out $OUTPUT -applyxfm -init $WANG2FUNC
  done
  
  #merge the mask from two hemisphere for selected ROI
  left=./${sub}/${ROI}_lh.nii.gz
  right=./${sub}/${ROI}_rh.nii.gz
#  output=./${sub}_${ROI}_combined.nii.gz
#  fslmaths $left -add $right $output 
  fslmaths $left -thr $THR -bin $left #take threshhold and then bin the data
  fslmaths $right -thr $THR -bin $right
done 

echo done

# fslview_deprecated  ${WANGINFUNC} ${TEMPLATE_bet}
