import nibabel as nib
import numpy as np
import os
import sys
import time

# What subject are you running
subject = sys.argv[1]
try:
    dataSource = sys.argv[2]  # could be neurosketch or realtime
    print("Using {} data".format(dataSource))
except:
    print("NO DATASOURCE ENTERED: Using original neurosketch data")
    dataSource = 'neurosketch'

def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))


if dataSource == "neurosketch":
    anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/{sub}_neurosketch_anat_mprage_brain.nii.gz"
elif dataSource == "realtime":
    anat = "$TO_BE_FILLED"
else:
    anat = "$TO_BE_FILLED"

    
# LOCATION TO SAVE SOME OUTPUT
outloc = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout"        
        
template = "/gpfs/milgram/scratch60/turk-browne/kp578/sandbox/{}_neurosketch_run1/0120.nii.gz".format(subject)
atlas = "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
stand = "/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_2mm_brain.nii.gz"
func2anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{}_neurosketch/data/nifti/realtime_preprocessed/func2anat.mat".format(subject)
anat2func = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{}_neurosketch/data/nifti/realtime_preprocessed/anat2func.mat".format(subject)
stan2anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{}_neurosketch/data/nifti/realtime_preprocessed/stan2anat.mat".format(subject)
stan2func = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{}_neurosketch/data/nifti/realtime_preprocessed/stan2func.mat".format(subject)
VCmask = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{}_neurosketch/data/nifti/realtime_preprocessed/VCmask.nii.gz".format(subject)

# which ROIs from harvard oxford atlas to remove from brain mask.
remove = [47, 48, 24]        

names4include = ["temporal pole", "STGa", "STGp", "ATGa", "ATGp", "ATGt", "ITa", "ITp", "ITt", "SPL", "LOs", "LOi", "PHCa", "PHCp", "TFUSa", "TFUSp", "TOFUS", "OFUS"]
include = [8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 22, 23, 34, 35, 37, 38, 39, 40]        
        
os.system("convert_xfm -omat {} -inverse {} > /dev/null".format(anat2func, func2anat))
Wait(anat2func)

os.system("flirt -in {} -ref {} -out {}/mask.nii.gz -init {} -applyxfm > /dev/null".format(anat.format(sub=subject), template, outloc, anat2func))
Wait("{}/mask.nii.gz".format(outloc))

# Create brain mask for searchlight, binarize it
mask = nib.load("{}/mask.nii.gz".format(outloc)).get_data()
mask = np.where(mask>0, 1, 0)
#mask[:,:,:] = 0
print("mask", mask.shape)

# Load in Harvard Oxford Atlas, find EVC regions, and then binarize.
HO = nib.load(atlas)
aff = HO.affine
HO = HO.get_data()
HO = np.where(np.isin(HO, remove), 1, 0)
#HO = np.where(np.isin(HO, include), 1, 0)
VC = nib.Nifti1Image(HO, aff)
VC.to_filename("{}/VC.nii.gz".format(outloc))

# The other half is getting from standard to anat, do this now
os.system("flirt -ref {} -in {} -out {}/temp.nii.gz -omat {} -cost corratio -dof 12 -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -interp trilinear  > /dev/null".format(anat.format(sub=subject), stand, outloc, stan2anat))
Wait(stan2anat)

# Then concatenate the halves to do the xfm in one single step
os.system("convert_xfm -concat {} -omat {} {} > /dev/null".format(anat2func, stan2func, stan2anat))
Wait(stan2func)

# Use it to transform the EVC ROI to functional template space
os.system("flirt -in {}/VC.nii.gz -ref {} -out {} -init {} -applyxfm > /dev/null".format(outloc, template, VCmask, stan2func))
Wait(VCmask)

os.system("fslmaths {} -bin {} > /dev/null".format(VCmask, VCmask))

time.sleep(5)

# Subtract EVC from mask
VC = nib.load(VCmask).get_data()
print("VC shape", VC.shape)
mask[VC > 0] = 0

maskVC = nib.Nifti1Image(mask, aff)
maskVC.to_filename("{}/{}_maskNoEVC.nii.gz".format(outloc, subject))

print("maskNoEVC saved")


