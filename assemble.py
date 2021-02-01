import nibabel as nib
import numpy as np
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

# What subject are you running
subject = sys.argv[1]



    
outloc = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout"
starttime = time.time()

def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        

__mask = nib.load("./wang2014/{}/roi1_lh.nii.gz".format(subject))
aff = __mask.affine
_mask = __mask.get_data()
#dimsize = runIm.header.get_zooms()

wangres = np.zeros(_mask.shape)
schaeferres = np.zeros(_mask.shape)

for roihemi in ["lh", "rh"]:
    for roinum in range(1, 26):
        mask = nib.load("./wang2014/{}/roi{}_{}.nii.gz".format(subject, roinum, roihemi)).get_data().astype(int)
        acc = np.load("./wang2014/{}/output/roi{}_{}.npy".format(subject, roinum, roihemi))
        wangres[mask==1] = acc

img = nib.Nifti1Image(wangres, aff)
nib.save(img, "./wang2014/{}/output/{}_wang.nii.gz".format(subject, subject))


for roinum in range(1, 300):
    mask = nib.load("./schaefer2018/{}/{}.nii.gz".format(subject, roinum)).get_data().astype(int)
    acc = np.load("./schaefer2018/{}/output/{}.npy".format(subject, roinum))
    schaeferres[mask==1] = acc

img = nib.Nifti1Image(schaeferres, aff)
nib.save(img, "./schaefer2018/{}/output/{}_schaefer.nii.gz".format(subject, subject))
print("done")
