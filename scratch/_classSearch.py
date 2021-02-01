import nibabel as nib
import numpy as np
import os
import sys
import time
import pandas as pd
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from sklearn.linear_model import LogisticRegression

# What subject are you running
subject = sys.argv[1]

try:
    radius = int(sys.argv[2])
    print("Using user-selected radius of {}".format(radius))
except:
    print("NO SL RADIUS ENTERED: Using radius of 3")
    radius = 3
try:
    dataSource = sys.argv[3]  # could be neurosketch or realtime
    print("Using {} data".format(dataSource))
except:
    print("NO DATASOURCE ENTERED: Using original neurosketch data")
    dataSource = 'neurosketch'
dist = int(sys.argv[4])
    
print("Running subject {}, with radius {}, and {} as a data source".format(subject, radius, dataSource))


# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# dataSource depending, there are a number of keywords to fill in: 
# ses: which day of data collection
# run: which run number on that day (single digit)
# phase: 12, 34, or 56
# sub: subject number
if dataSource == "neurosketch":
    funcdata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/realtime_preprocessed/{sub}_neurosketch_recognition_run_{run}.nii.gz"
    metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
    anat = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/{sub}_neurosketch/data/nifti/{sub}_neurosketch_anat_mprage_brain.nii.gz"
elif dataSource == "realtime":
    funcdata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/nifti/{sub}_functional.nii.gz"
    metadata = "/gpfs/milgram/project/turk-browne/projects/rtcloud_kp/subjects/{sub}/ses{ses}_recognition/run0{run}/{sub}_0{run}_preprocessed_behavData.csv"
    anat = "$TO_BE_FILLED"
else:
    funcdata = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout/feat/{sub}_pre.nii.gz"
    metadata = "/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/data/features/recog/metadata_{sub}_V1_{phase}.csv"
    anat = "$TO_BE_FILLED"
    
outloc = "/gpfs/milgram/project/turk-browne/projects/rtTest/searchout"
starttime = time.time()

def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def normalize(X):
    X = X - X.mean(3)
    return X

def Class(data, sl_mask, myrad, bcvar):
    metas = bcvar[0]
    data4d = data[0]
    data4d = data4d.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], data4d.shape[3]).T
    data4d = data4d.reshape(6, 80, sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2])
#    data4d = data4d[40:57, 40:57, 40:57]
#    data4d = data4d.reshape(4913, data4d.shape[3]).T
#    data4d = data4d.reshape(6, 80, 4913)
    
    accs = []
    for run in range(6):
        testX = data4d[run]
        testY = metas[run]
        trainX = data4d[np.arange(6) != run]
        trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1], -1)
        trainY = []
        for meta in range(6):
            if meta != run:
                trainY.extend(metas[run])
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
                
        # Monitor progress by printing accuracy (only useful if you're running a test set)
        acc = clf.score(testX, testY)
        accs.append(acc)
    
    return np.mean(accs)


phasedict = dict(zip([1,2,3,4,5,6],["12", "12", "34", "34", "56", "56"]))
imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}


Wait("{}/{}_maskNoEVC.nii.gz".format(outloc, subject))
mask = nib.load("{}/{}_maskNoEVC.nii.gz".format(outloc, subject)).get_data()
mask[:,:,:] = 0
mask[40:40+dist, 40:40+dist, 40:40+dist] = 1

# Compile preprocessed data and corresponding indices
metas = []

for run in range(1, 7):
    print(run, end='--')
    # retrieve from the dictionary which phase it is, assign the session
    phase = phasedict[run]
    ses = 1
    
    # Build the path for the preprocessed functional data
    this4d = funcdata.format(ses=ses, run=run, phase=phase, sub=subject)
    
    # Read in the metadata, and reduce it to only the TR values from this run, add to a list
    thismeta = pd.read_csv(metadata.format(ses=ses, run=run, phase=phase, sub=subject))
    if dataSource == "neurosketch":
        _run = 1 if run % 2 == 0 else 2
    else:
        _run = run
    thismeta = thismeta[thismeta['run_num'] == int(_run)]
    
    if dataSource == "realtime":
        TR_num = list(thismeta.TR.astype(int))
        labels = list(thismeta.Item)
        labels = [imcodeDict[label] for label in labels]
    else:
        TR_num = list(thismeta.TR_num.astype(int))
        labels = list(thismeta.label)
    
    print("LENGTH OF TR: {}".format(len(TR_num)))
    # Load the functional data
    runIm = nib.load(this4d)
    affine_mat = runIm.affine
    runImDat = runIm.get_data()
    
    # Use the TR numbers to select the correct features
    features = [runImDat[:,:,:,n+3] for n in TR_num]
    features = np.array(features)
    featmean = features.mean(3)[..., None]
    features = features - featmean
    features = np.expand_dims(features, 0)
    
    # Append both so we can use it later
    metas.append(labels)
    runs = features if run == 1 else np.concatenate((runs, features))

dimsize = runIm.header.get_zooms()


data = []
# Preset the variables
if rank == 0:
    _data = runs.reshape(runs.shape[0] * runs.shape[1], runs.shape[2], runs.shape[3], runs.shape[4]).T
    print(_data.shape)
    data.append(_data)
    print("shape of data: {}".format(_data.shape))
else:
    _data = None
    data.append(_data)
    
bcvar = [metas]
                 
# say some things about the mask.
print('mask dimensions: {}'. format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))

sl_rad = radius
max_blk_edge = 5
pool_size = 1



# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute(data, mask)
sl.broadcast(bcvar)
slstart = time.time()
sl_result = sl.run_searchlight(Class)

#result = Class(data, np.zeros((5,5,5)), 2, bcvar)

SL = time.time() - slstart
tot = time.time() - starttime
print('total time: {}, searchlight time: {}'.format(tot, SL))
'''
# Only save the data if this is the first core
if rank == 0:
    output = ('{}/{}_r{}.npy'.format(outloc, subject, radius))
    #np.save(output, sl_result)

    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0  # If there are nans we want this

    # Save the volume
    sl_nii = nib.Nifti1Image(sl_result.astype('double'), affine_mat)
    #hdr = sl_nii.header
    #hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
    #nib.save(sl_nii, '{}.nii.gz'.format(output.split('.npy')[0]))  # Save
'''

