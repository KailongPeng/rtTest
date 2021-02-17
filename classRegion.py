'''
from the main directory, run batchRegions.sh, it will run the script classRegion.sh, which is just a feeder for classRegion.py for all ROI/parcels across both wang and schaefer.

classRegion.py simply runs a runwise cross-validated classifier across the runs of recognition data, then stores the average accuracy of the ROI it was assigned in an numpy array. 
This is stored within the subject specific folder (e.g. wang2014/0111171/output/roi25_rh.npy )

input:
    1 subject: which subject
    2 dataloc: neurosketch or realtime
    3 roiloc: schaefer2018 or wang2014
    4 roinum: number of rois you want
    5 roihemi: which hemisphere

'''
import nibabel as nib
import numpy as np
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

# What subject are you running
subject = sys.argv[1]

try:
    roiloc = str(sys.argv[3])
    print("Using user-selected roi location: {}".format(roiloc))
except:
    print("NO ROI LOCATION ENTERED: Using radius of wang2014")
    roiloc = "wang2014"

try:
    dataSource = sys.argv[2]  # could be neurosketch or realtime
    print("Using {} data".format(dataSource))
except:
    print("NO DATASOURCE ENTERED: Using original neurosketch data")
    dataSource = 'neurosketch'

try:
    roinum = str(sys.argv[4]) if roiloc == "schaefer2018" else "roi{}".format(str(sys.argv[4])) 
    print("running for roi #{} in {}".format(str(sys.argv[4]), roiloc))
except:
    print("NO ROI SPECIFIED: Using roi number 1")
    roinum="1"

if roiloc == "wang2014":
    try:
        roihemi = "_{}".format(str(sys.argv[5]))
        print("Since this is wang2014, we need a hemisphere, in this case {}".format(str(sys.argv[5])))
    except:
        print("this is wang 2014, so we need a hemisphere, but one was not specified")
        assert 1 == 2
else:
    roihemi=""

print("Running subject {}, with {} as a data source, {} roi #{} {}".format(subject, dataSource, roiloc, roinum, roihemi))

workDir="/gpfs/milgram/project/turk-browne/projects/rtTest/"
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

def Class(data, bcvar):
    metas = bcvar[0]
    data4d = data[0]
    print(data4d.shape)

    accs = []
    for run in range(6):
        testX = data4d[run]
        testY = metas[run]
        trainX = data4d[np.arange(6) != run]
        trainX = trainX.reshape(trainX.shape[0]*trainX.shape[1], -1)
        trainY = []
        for meta in range(6):
            if meta != run:
                trainY.extend(metas[meta])
        clf = LogisticRegression(penalty='l2',C=1, solver='lbfgs', max_iter=1000, 
                                 multi_class='multinomial').fit(trainX, trainY)
                
        # Monitor progress by printing accuracy (only useful if you're running a test set)
        acc = clf.score(testX, testY)
        accs.append(acc)
    
    return np.mean(accs)


phasedict = dict(zip([1,2,3,4,5,6],["12", "12", "34", "34", "56", "56"]))
imcodeDict={"A": "bed", "B": "Chair", "C": "table", "D": "bench"}

mask = nib.load(workDir+"{}/{}/{}{}.nii.gz".format(roiloc, subject, roinum, roihemi)).get_data()
mask = mask.astype(int)
# say some things about the mask.
print('mask dimensions: {}'. format(mask.shape))
print('number of voxels in mask: {}'.format(np.sum(mask)))


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
    features = [runImDat[:,:,:,n+3] for n in TR_num] #axis0 is time/repetition
    features = np.array(features)
    features = features[:, mask==1]
    print("shape of features", features.shape, "shape of mask", mask.shape)
    # featmean = features.mean(1)[..., None] # equivalent to featmean = np.mean(features,axis=1).reshape(-1,1)
    features = features - features.mean(0)
    features = np.expand_dims(features, 0)
    
    # Append both so we can use it later
    metas.append(labels)
    runs = features if run == 1 else np.concatenate((runs, features))

dimsize = runIm.header.get_zooms()

data = []
# Preset the variables
print("Runs shape", runs.shape)
_data = runs
print(_data.shape)
data.append(_data)
print("shape of data: {}".format(_data.shape))
    
bcvar = [metas]
                 
# Distribute the information to the searchlights (preparing it to run)
slstart = time.time()
sl_result = Class(data, bcvar)
print("results of classifier: {}, type: {}".format(sl_result, type(sl_result)))

SL = time.time() - slstart
tot = time.time() - starttime
print('total time: {}, searchlight time: {}'.format(tot, SL))

outfile = workDir+"{}/{}/output/{}{}.npy".format(roiloc, subject, roinum, roihemi)
print(outfile)
np.save(outfile, np.array(sl_result))
