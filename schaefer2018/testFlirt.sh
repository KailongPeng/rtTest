sub=1206162 #0125171 bad orientation 0120172 #0115174 bad orientation #0119171 missing part of brain #0120171 missing part of brain #0112172 #0119173 missing part of brain #1206162 #1206161 #1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172
# bad orientation 0125171 0115174
STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz

THR=10
ROIpath=/gpfs/milgram/scratch/turk-browne/tsy6/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI
WANG2FUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/wang2func.mat
TEMPLATE="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz"
TEMPLATE_bet=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz

WANGINFUNC=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/wanginfunc.nii.gz

# flirt -ref $TEMPLATE_bet -in $STAND -omat $WANG2FUNC -out $WANGINFUNC
# fslview_deprecated  ${WANGINFUNC} ${TEMPLATE}
fslview_deprecated  ${WANGINFUNC} ${TEMPLATE_bet}









# fslview_deprecated sumMask_38.nii.gz $STAND


# bet ${TEMPLATE} ${TEMPLATE_bet}
# bet ${TEMPLATE} ${TEMPLATE_bet} # missing part of brain
# bet -f 0 ${TEMPLATE} ${TEMPLATE_bet} # missing part of brain
# bet -f 1 ${TEMPLATE} ${TEMPLATE_bet} # missing part of brain
# bet -g -1 ${TEMPLATE} ${TEMPLATE_bet}
# bet -g 1 ${TEMPLATE} ${TEMPLATE_bet}








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
  



sub=0120172 #1121161 not working
FUNC_bet=/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz
topN=/gpfs/milgram/project/turk-browne/projects/rtTest/schaefer2018/${sub}/output/top38mask.nii.gz
fslview_deprecated ${FUNC_bet} ${topN}