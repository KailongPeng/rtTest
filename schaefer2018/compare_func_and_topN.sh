
sub="0110172"
FUNC="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz"
FUNC_bet="/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz"
topN="/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/output/top50mask.nii.gz"

fslview_deprecated  ${FUNC} ${topN}






# 0110171  0112174  0118172  0120172  0125172  1206161
# 0110172  0113171  0119171  0120173  1121161  1206162
# 0111171  0115172  0119172  0123171  1130161  1206163
# 0112171  0115174  0119173  0123173  1201161  1207162
# 0112172  0117171  0119174  0124171  1202161  
# 0112173  0118171  0120171  0125171  1203161
sub="1203161" #"1206163" #"0110172"
FUNC="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/${sub}_neurosketch/data/nifti/realtime_preprocessed/${sub}_neurosketch_recognition_run_1.nii.gz"
FUNC_bet="/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/neurosketch_recognition_run_1_bet.nii.gz"
STANDinFUNC="/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/${sub}/wanginfunc.nii.gz"
topN="/gpfs/milgram/project/turk-browne/projects/rtTest/schaefer2018/${sub}/output/top300mask.nii.gz"

fslview_deprecated  ${FUNC} ${topN} ${STANDinFUNC} ${FUNC_bet}

