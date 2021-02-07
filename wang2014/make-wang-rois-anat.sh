#!/bin/bash
#SBATCH --partition=short   
#SBATCH --job-name=sMasks
#SBATCH --time=40:00
#SBATCH --output=wangMask-%j.out
#SBATCH --mem=2g

set -e #stop immediately encountering error

sub=$1
mkdir -p ./${sub} # save the output files in the current folder


STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz
THR=10
ROIS="roi1 roi2 roi3 roi4 roi5 roi6 roi7 roi8 roi9 roi10 roi11 roi12 roi13 roi14 roi15 roi16 roi17 roi18 roi19 roi20 roi21 roi22 roi23 roi24 roi25"

ROIpath=/gpfs/milgram/project/turk-browne/shared_resources/atlases/ProbAtlas_v4/subj_vol_all

#register deskulled roi to individual subject t1
WANG2ANAT=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/wang2anat.mat
ANAT2FUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/anat2func.mat
WANG2FUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/wang2func.mat

ANAT_bet=/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/${sub}_neurosketch_anat_mprage_brain.nii.gz
FUNC=/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz
FUNC_bet=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz
bet ${FUNC} ${FUNC_bet}

WANGinANAT=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/WANGinANAT.nii.gz
WANGinFUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/WANGinFUNC.nii.gz
ANATinFUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/ANATinFUNC.nii.gz

# wang to anat
flirt -ref $ANAT_bet -in $STAND -omat $WANG2ANAT -out $WANGinANAT

# anat to func
# flirt -ref $FUNC_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC -dof 6
flirt -ref $FUNC_bet -in $ANAT_bet -omat $ANAT2FUNC -out $ANATinFUNC

# apply anat to func on wang_in_anat
flirt -ref $FUNC_bet -in $WANGinANAT  -out $WANGinFUNC -applyxfm -init $ANAT2FUNC

# combine wang2anat and anat2func to wang2func
# convert_xfm -omat AtoC.mat -concat BtoC.mat AtoB.mat
convert_xfm -omat $WANG2FUNC -concat $ANAT2FUNC $WANG2ANAT


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

