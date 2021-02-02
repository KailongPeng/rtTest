import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}") 

import numpy as np
import nibabel as nib
import os
import sys
import time
from nilearn.image import new_img_like
# sys.path.append('/gpfs/milgram/project/turk-browne/users/kp578/realtime/rt-cloud/')

def print_orien(example_file):
    img = nib.load(example_file)
    # Here is the affine (to two digits decimal precision):
    np.set_printoptions(precision=2, suppress=True)
    # print(f"img.affine={img.affine}")
    # What are the orientations of the voxel axes here?
    # Nibabel has a routine to tell you, called aff2axcodes.
    orientation = nib.aff2axcodes(img.affine)
    print(f"orientation of {example_file} = {orientation}")
    return nib.aff2axcodes(img.affine)

def orien_trans(stand,func,stand_funcOrien_fileName):
    orientation_func = print_orien(func)
    orientation_stand = print_orien(stand)
    
    # convert stand brain to functional space
    stand_data = nib.load(stand).get_data()
    ornt_transform = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(orientation_stand),
        nib.orientations.axcodes2ornt(orientation_func))
    stand_funcOrien = nib.orientations.apply_orientation(stand_data, ornt_transform)
    correct_object = new_img_like(func,stand_funcOrien, copy_header=True)
    correct_object.to_filename(stand_funcOrien_fileName)


# func = os.path.join("/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/0110171", 'neurosketch_recognition_run_1_bet.nii.gz')
# stand="/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz"
# stand_funcOrien="/gpfs/milgram/project/turk-browne/projects/rtTest/wang2014/0110171/stand_funcOrien.nii.gz"
# orien_trans(stand,func,stand_funcOrien)
print(f"sys.argv[1]={sys.argv[1]}, sys.argv[2]={sys.argv[2]}, sys.argv[3]={sys.argv[3]}")
orien_trans(sys.argv[1],
sys.argv[2],
sys.argv[3])