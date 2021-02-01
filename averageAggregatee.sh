#!/usr/bin/env bash

# bash code to get the average map of all available subjects: 
# analysis code for the aggregate.py for rtTest folder from Jeff's mask selection code

# take the average of the top1 accuracy and save the average file as a nii.gz file for fsl to view


subjects="1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172"


roiloc=wang2014
for sub in $subjects;do
    for num in {1..25};do
        sbatch batch_flirt.sh $sub $num $roiloc
        # # convert the topMasks into standard space
        #     # convert wang2func.mat to func2wang.mat
        # WANG2FUNC=./wang2014/${sub}/wang2func.mat
        # FUNC2WANG=./wang2014/${sub}/func2wang.mat
        # convert_xfm -omat ${FUNC2WANG} -inverse ${WANG2FUNC}
        #     # convert individual space to stand space
        # INPUT=./wang2014/${sub}/output/top${num}mask.nii.gz
        # OUTPUT=./wang2014/${sub}/output/STAND_top${num}mask.nii.gz
        # echo flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
        # flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
    done
done



roiloc=schaefer2018
for sub in $subjects;do
    for num in {1..300};do
        sbatch batch_flirt.sh $sub $num $roiloc
            # convert individual space to stand space
        # INPUT=./schaefer2018/${sub}/output/top${num}mask.nii.gz
        # OUTPUT=./schaefer2018/${sub}/output/STAND_top${num}mask.nii.gz
        # echo flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
        # flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
    done
done


# # python 
# AddUpAverage.py
# to add the top performed ROIs together to visualize the masks

# di="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/"
# from glob import glob
# from subprocess import call
# import nib,os
# subs=glob(f"{di}[0,1]*_neurosketch")
# subs=[sub.split("/")[-1].split("_")[0] for sub in subs]

# testDir='/gpfs/milgram/project/turk-browne/projects/rtTest/'
# subjects=subs


# for curr_roi in range(1,1+25):
#     mask=[]
#     command="fslmaths "
#     for sub in subjects:
#         file=f"./wang2014/{sub}/output/STAND_top{curr_roi}mask.nii.gz"
#         path="/gpfs/milgram/project/turk-browne/projects/rtTest/"
#         if os.path.exists(path+file):
#             command=command+file+" -add "
#     command=command[:-6]
#     command=command+f' ./wang2014/sumMask_{curr_roi}.nii.gz'
#     if curr_roi==2:
#         print(command)
#         call(command,shell=True)


# for curr_roi in range(1,1+300):
#     mask=[]
#     command="fslmaths "
#     for sub in subjects:
#         file=f"./schaefer2018/{sub}/output/STAND_top{curr_roi}mask.nii.gz"
#         path="/gpfs/milgram/project/turk-browne/projects/rtTest/"
#         if os.path.exists(path+file):
#             command=command+file+" -add "
#     command=command[:-6]
#     command=command+f' ./schaefer2018/sumMask_{curr_roi}.nii.gz'
#     if curr_roi==2:
#         print(command)
#         call(command,shell=True)


# fslview_deprecated ./schaefer2018/sumMask_38.nii.gz /gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz 