'''
purpose:
    according to the given number of ROIs N, pick up the top N ROIs accuracy and combine them for a combined mask and retrain the model and getting result

steps:
    load accuracy for all the ROIs for given subject
    pick up the top N ROIs
    combine these top N masks
    retrain the model and get the accuracy.

steps modified for a unified mask rank for all subjects: uniMaskRank_
    load accuracy for all the ROIs for all subjects
    for each ROI, average the performance from each subject and get a single number for each ROI
    pick up the top N ROIs
    combine these top N masks
    retrain the model and get the accuracy.

'''
'''
you could try to see whether combining parcels improves performance. 
That's going to be the most important bit, because we'll want to decide on a tradeoff between number of voxels and accuracy. 
The script of interest here is aggregate.sh which is just a feeder for aggregate.py. 
This will use the .npy outputs of classRegion.py to select and merge the top N ROIs/parcels, and will return the list of ROI names, the number of voxels, and the cross-validated classifier accuracy 
in this newly combined larger mask. An example run of this is as follows:
sbatch aggregate.sh 0111171 neurosketch schaefer2018 15
'''
import numpy as np
import nibabel as nib
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression

# What subject are you running
'''
Takes args (in order):
    subject (e.g. 0111171)
    dataSource (e.g. neurosketch, but also realtime)
    roiloc (wang2014 or schaefer2018)
    N (the number of parcels or ROIs to start with)
'''
subject = sys.argv[1]
N = int(sys.argv[4])

try:
    roiloc = str(sys.argv[3])
    print("Using user-selected roi location: {}".format(roiloc))
except:
    print("NO ROI LOCATION ENTERED: Using roi location of wang2014")
    roiloc = "wang2014"

try:
    dataSource = sys.argv[2]  # could be neurosketch or realtime
    print("Using {} data".format(dataSource))
except:
    print("NO DATASOURCE ENTERED: Using original neurosketch data")
    dataSource = 'neurosketch'

print("Running subject {}, with {} as a data source, {}, starting with {} ROIs".format(subject, dataSource, roiloc, N))

tag="GM"

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

workingDir="/gpfs/milgram/project/turk-browne/projects/rtTest/"
starttime = time.time()
# '1201161', '1121161','0115172','0112174' #these subject have nothing in output folder
subjects_correctly_aligned=['1206161','0119173','1206162','1130161','1206163','0120171','0111171','1202161','0125172','0110172','0123173','0120173','0110171','0119172','0124171','0123171','1203161','0118172','0118171','0112171','1207162','0117171','0119174','0112173','0112172']
if roiloc == "schaefer2018":
    RESULT=np.zeros((len(subjects_correctly_aligned),300))
    topN = []
    for ii,sub in enumerate(subjects_correctly_aligned):
        outloc = workingDir+"/{}/{}/output".format(roiloc, sub)
        for roinum in range(1,301):
            try: # 这里之所以要“try”是因为有的Schaefer ROI在被GM mask之后的voxel数目变成了0
                result = np.load(f"{outloc}/{roinum}_{tag}.npy")
                RESULT[ii,roinum-1]=result
            except:
                pass
            # RESULT = result if roinum == 1 else np.vstack((RESULT, result))
    RESULT = np.nanmean(RESULT,axis=0)
    print(f"RESULT.shape={RESULT.shape}")
    RESULTix = RESULT[:].argsort()[-N:]
    for idx in RESULTix:
        topN.append(f"{tag}_{idx+1}.nii.gz")
        print(topN[-1])
else:
    RESULT_all=[]
    topN = []
    for ii,sub in enumerate(subjects_correctly_aligned):
        outloc = workingDir+"/{}/{}/output".format(roiloc, sub)
        for hemi in ["lh", "rh"]:
            for roinum in range(1, 26):
                result = np.load(f"{outloc}/roi{roinum}_{hemi}_{tag}.npy")
                # result = np.load("{}/roi{}_{}.npy".format(outloc, roinum, hemi))
                # outfile = workDir+"{}/{}/output/tag2_{}{}.npy".format(roiloc, subject, roinum, roihemi)
                Result = result if roinum == 1 else np.vstack((Result, result))
            RESULT = Result if hemi == "lh" else np.hstack((RESULT, Result))
        RESULT_all.append(RESULT)

    RESULT_all=np.asarray(RESULT_all)
    print(f"RESULT_all.shape={RESULT_all.shape}")
    RESULT_all=np.mean(RESULT_all,axis=0)
    print(f"RESULT_all.shape={RESULT_all.shape}")
    RESULT1d = RESULT.flatten()
    RESULTix = RESULT1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(RESULTix, RESULT.shape)

    # Check that we got the largest values.
    for x, y, in zip(x_idx, y_idx):
        print(x,y)
        if y == 0:
            topN.append(f"{tag}_roi{x+1}_lh.nii.gz")
        else:
            topN.append(f"{tag}_roi{x+1}_rh.nii.gz")
        print(topN[-1])


def Wait(waitfor, delay=1):
    while not os.path.exists(waitfor):
        time.sleep(delay)
        print('waiting for {}'.format(waitfor))
        
def normalize(X):
    from scipy.stats import zscore
    # X = X - X.mean(0)
    X = zscore(X, axis=0)
    X[np.isnan(X)]=0
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

for pn, parc in enumerate(topN):
    _mask = nib.load(workingDir+f"/{roiloc}/{subject}/{parc}")
    aff = _mask.affine
    _mask = _mask.get_data()
    _mask = _mask.astype(int)
    # say some things about the mask.
    mask = _mask if pn == 0 else mask + _mask
    mask[mask>0] = 1
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
    features = [runImDat[:,:,:,n+3] for n in TR_num]
    features = np.array(features)
    features = features[:, mask==1]
    print("shape of features", features.shape, "shape of mask", mask.shape)
    # featmean = features.mean(1)[..., None]
    # features = features - featmean
    # features = features - features.mean(0)
    features = normalize(features)
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

# SAVE accuracy
outfile = workingDir+f"/{roiloc}/{subject}/output/uniMaskRank_{tag}_top{N}.npy"
print(f"sl_result path={outfile}")
np.save(outfile, np.array(sl_result))
# SAVE mask
savemask = nib.Nifti1Image(mask, affine=aff)
nib.save(savemask, workingDir+f"/{roiloc}/{subject}/output/uniMaskRank_{tag}_top{N}mask.nii.gz")
# SAVE roilist, nvox
ROILIST = [r for r in topN]
ROILIST.append(np.sum(mask))
ROILIST = pd.DataFrame(ROILIST)
ROILIST.to_csv(workingDir+f"/{roiloc}/{subject}/output/uniMaskRank_{tag}_top{N}.csv")





def plot():
    # code to load and compare the result of above:
    from glob import glob
    import numpy as np
    # di="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/"
    # subs=glob(f"{di}[0,1]*_neurosketch")
    # subs=[sub.split("/")[-1].split("_")[0] for sub in subs]
    # subjects=""
    # for sub in subs:
    #     subjects=subjects+sub+" "
        
        
    testDir='/gpfs/milgram/project/turk-browne/projects/rtTest/'
    subjects_correctly_aligned=['1206161','0119173','1206162','1130161','1206163','0120171','0111171','1202161','0125172','0110172','0123173','0120173','0110171','0119172','0124171','0123171','1203161','0118172','0118171','0112171','1207162','0117171','0119174','0112173','0112172']
    subs=subjects_correctly_aligned
    subjects=subs #["0110171", "0110172", "0111171"]
    hemis=["lh", "rh"]
    tag="GM"
    wangAcc=np.zeros((50,len(subs)))
    # roiloc="wang2014"
    # for sub_i,sub in enumerate(subjects):
    #     for num in range(1,51):
    #         # try:
    #         wangAcc[num-1,sub_i]=np.load(f"{testDir}{roiloc}/{sub}/output/uniMaskRank_{tag}_top{num}.npy")
    #         # print(f"{roiloc} {sub} {num} ROIs acc={wangAcc[num-1,sub_i]}")
    #         # except:
    #         #     pass

    schaeferAcc=np.zeros((300,len(subs)))
    roiloc="schaefer2018"
    for sub_i,sub in enumerate(subjects):
        for num in range(1,301):
            try:
                schaeferAcc[num-1,sub_i]=np.load(f"{testDir}{roiloc}/{sub}/output/uniMaskRank_{tag}_top{num}.npy")
            # print(f"{roiloc} {sub} {num} ROIs acc={schaeferAcc[num-1,sub_i]}")
            except:
                pass


    wangAcc=wangAcc[:,wangAcc[0]!=0]
    schaeferAcc=schaeferAcc[:,schaeferAcc[0]!=0]
    schaeferAcc[schaeferAcc==0]=None

    import matplotlib.pyplot as plt
    plt.plot(np.nanmean(wangAcc,axis=1))
    plt.plot(np.nanmean(schaeferAcc,axis=1))


    for i in range(schaeferAcc.shape[0]):
        plt.scatter([i]*schaeferAcc.shape[1],schaeferAcc[i],c='g',s=2)
    for i in range(wangAcc.shape[0]):
        plt.scatter([i]*wangAcc.shape[1],wangAcc[i],c='b',s=2)
        
    plt.xlabel("number of ROIs")
    plt.ylabel("accuracy")
    # plt.savefig('SummaryAccuracy.png')


    plt.figure()
    plt.plot(np.mean(schaeferAcc,axis=1))
    plt.plot(np.mean(wangAcc,axis=1))


    schaeferAcc_mean=np.mean(schaeferAcc,axis=1)
    bestID=np.where(schaeferAcc_mean==np.nanmax(schaeferAcc_mean))[0][0]
    _=plt.figure()
    for i in range(schaeferAcc.shape[0]):
        plt.scatter([i]*schaeferAcc.shape[1],schaeferAcc[i],c='g',s=2)
    plt.plot(np.arange(schaeferAcc.shape[0]),np.nanmean(schaeferAcc,axis=1))
    plt.ylim([0.19,0.36])
    plt.plot([bestID]*10,np.arange(0.19,0.36,(0.36-0.19)/10))


    plt.plot(np.arange(schaeferAcc.shape[0]),np.nanmean(schaeferAcc,axis=1))
    plt.plot([bestID]*10,np.arange(0.255,0.264,(0.264-0.255)/10))
    plt.title(f"bestID={bestID}")
