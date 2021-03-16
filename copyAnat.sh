


subjects="1206161 0119173 1206162 1130161 1206163 0120171 0111171 1202161 0125172 0110172 0123173 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0117171 0119174 0112173 0112172" #these subjects are done with the batchRegions code
rttPath="/gpfs/milgram/project/turk-browne/projects/rtTest/"
for sub in $subjects;do
    ANAT_bet=/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/${sub}_neurosketch_anat_mprage_brain.nii.gz
    mkdir -p ${rttPath}/subjects/${sub}/anat/

    # copy anat to working folder
    cp $ANAT_bet ${rttPath}/subjects/${sub}/anat/anat.nii.gz 

    # change anat orientation to standard 
    fslreorient2std ${rttPath}/subjects/${sub}/anat/anat.nii.gz  ${rttPath}/subjects/${sub}/anat/anat_reorien.nii.gz 
    # fslorient -getorient ${rttPath}/subjects/${sub}/anat/anat_reorien.nii.gz 
done
