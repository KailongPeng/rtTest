'''
purpose:
    according to the given number of ROIs N, pick up the top N ROIs accuracy and combine them for a combined mask and retrain the model and getting result

steps:
    load accuracy for all the ROIs for given subject
    pick up the top N ROIs
    combine these top N masks
    retrain the model and get the accuracy.

    get the N combinations of N-1 ROIs
    retrain the model and get the accuracy for these N combinations

    get the N-1 combinations of N-2 ROIs
    retrain the model and get the accuracy for these N-1 combinations

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
import itertools
from tqdm import tqdm
import pickle
import subprocess
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# What subject are you running
'''
Takes args (in order):
    subject (e.g. 0111171)
    dataSource (e.g. neurosketch, but also realtime)
    roiloc (wang2014 or schaefer2018)
    N (the number of parcels or ROIs to start with)
'''
subject = sys.argv[1] # "0111171"
N = int(sys.argv[4]) # 38

try:
    roiloc = str(sys.argv[3]) #"schaefer2018"
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
    
starttime = time.time()

outloc = "./{}/{}/output".format(roiloc, subject)

if roiloc == "schaefer2018":
    topN = []
    for roinum in range(1,301):
        result = np.load("{}/{}.npy".format(outloc, roinum))
        RESULT = result if roinum == 1 else np.vstack((RESULT, result))
    # select N ROIs
    RESULTix = RESULT[:,0].argsort()[-N:] # argsort from small to big
    for idx in RESULTix:
        topN.append("{}.nii.gz".format(idx+1))
        print(topN[-1])
else:
    topN = []
    for hemi in ["lh", "rh"]:
        for roinum in range(1, 26):
            result = np.load("{}/roi{}_{}.npy".format(outloc, roinum, hemi))
            Result = result if roinum == 1 else np.vstack((Result, result))
        RESULT = Result if hemi == "lh" else np.hstack((RESULT, Result))

    RESULT1d = RESULT.flatten()
    RESULTix = RESULT1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(RESULTix, RESULT.shape)

    # Check that we got the largest values.
    for x, y, in zip(x_idx, y_idx):
        print(x,y)
        if y == 0:
            topN.append("roi{}_lh.nii.gz".format(x+1))
        else:
            topN.append("roi{}_rh.nii.gz".format(x+1))
        print(topN[-1])


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
    for run in tqdm(range(6)):
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

def getMask(topN):
    for pn, parc in enumerate(topN):
        _mask = nib.load("./{}/{}/{}".format(roiloc, subject, parc))
        aff = _mask.affine
        _mask = _mask.get_data()
        _mask = _mask.astype(int)
        # say some things about the mask.
        mask = _mask if pn == 0 else mask + _mask
        mask[mask>0] = 1
    return mask

mask=getMask(topN)

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
    # features = features[:, mask==1]
    print("shape of features", features.shape, "shape of mask", mask.shape)
    featmean = features.mean(1).mean(1).mean(1)[..., None,None,None] #features.mean(1)[..., None]
    features = features - featmean
    features = np.expand_dims(features, 0)
    
    # Append both so we can use it later
    metas.append(labels)
    runs = features if run == 1 else np.concatenate((runs, features))

dimsize = runIm.header.get_zooms()


# Preset the variables
print("Runs shape", runs.shape)
bcvar = [metas]
                 
# # Distribute the information to the searchlights (preparing it to run)
# _runs = [runs[:,:,mask==1]]
# print("Runs shape", _runs[0].shape)
# slstart = time.time()
# sl_result = Class(_runs, bcvar)
# print("results of classifier: {}, type: {}".format(sl_result, type(sl_result)))
# SL = time.time() - slstart
# tot = time.time() - starttime
# print('total time: {}, searchlight time: {}'.format(tot, SL))

def wait(tmpFile):
    while not os.path.exists(tmpFile+'_result.npy'):
        time.sleep(5)
        print("waiting\n")
    return np.load(tmpFile+'_result.npy')

# ./tmp/0125171_40_schaefer2018_neurosketch_39.pkl
if os.path.exists(f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_{1}.pkl"):
    print(f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_1.pkl exists")
    raise Exception('runned or running')

# N-1
def next(topN):
    if len(topN)==1:
        return None
    else:
        try:
            allpairs = itertools.combinations(topN,len(topN)-1)
            topNs=[]
            sl_results=[]
            tmpFiles=[]
            for i,_topN in tqdm(enumerate(allpairs)):
                topNs.append(_topN)
                _mask=getMask(_topN)
                # np.save(tmpFile,_mask)
                _runs = [runs[:,:,_mask==1]]

                print('mask dimensions: {}'. format(_mask.shape))
                print('number of voxels in mask: {}'.format(np.sum(_mask)))
                print("Runs shape", _runs[0].shape)

                tmpFile=f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)}_{i}"
                save_obj([_runs,bcvar], tmpFile)
                tmpFiles.append(tmpFile)
                if not os.path.exists(tmpFile+'_result.npy'):
                    subprocess.Popen(['squeue -u kp578 | wc -l > squeue.txt'],shell=True) # sl_result = Class(_runs, bcvar)
                    numberOfJobsRunning = int(open("squeue.txt", "r").read())
                    while numberOfJobsRunning > 100:
                        time.sleep(30)
                        print("waiting 30")
                        subprocess.Popen(['squeue -u kp578 | wc -l > squeue.txt'],shell=True) # sl_result = Class(_runs, bcvar)
                        numberOfJobsRunning = int(open("squeue.txt", "r").read())
                    proc = subprocess.Popen([f'sbatch class.sh {tmpFile}'],shell=True) # sl_result = Class(_runs, bcvar)
            sl_results=[]
            for tmpFile in tmpFiles:
                sl_result=wait(tmpFile)
                sl_results.append(sl_result)
            print(f"sl_results={sl_results}")
            print(f"max(sl_results)=={max(sl_results)}")
            maxID=np.where(sl_results==max(sl_results))[0][0]
            save_obj({"subject":subject,
            "startFromN":N,
            "currNumberOfROI":len(topN)-1,
            "bestAcc":max(sl_results),
            "bestROIs":topNs[maxID]},
            f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}"
            )
            print(f"bestAccFor {len(topN)-1} = ./tmp/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}")
            tmpFiles=next(topNs[maxID])
        except:
            return tmpFiles
tmpFiles=next(topN)



def Plot():
    from glob import glob
    import matplotlib.pyplot as plt

    roiloc="schaefer2018"
    dataSource="neurosketch"
    subjects=glob("./wang2014/[0,1]*")
    subjects=[subject.split("/")[-1] for subject in subjects]

    GreedyBestAcc=np.zeros((len(subjects),40))
    for ii,subject in enumerate(subjects):            
        for len_topN_1 in range(40,0,-1):
            Wait(f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_{len(topN)-1}.pkl")
            di = load_obj(f"./tmp/{subject}_{N}_{roiloc}_{dataSource}_{len_topN_1}")
            GreedyBestAcc[ii,len_topN_1-1] = di['bestAcc']

    plt.scatter(GreedyBestAcc,c='b',s=2)
        
    plt.xlabel("number of ROIs")
    plt.ylabel("accuracy")
    # plt.savefig('SummaryAccuracy.png')

# #SAVE accuracy
# outfile = "./{}/{}/output/top{}.npy".format(roiloc, subject, N)
# np.save(outfile, np.array(sl_result))
# #SAVE mask
# savemask = nib.Nifti1Image(mask, affine=aff)
# nib.save(savemask, "./{}/{}/output/top{}mask.nii.gz".format(roiloc, subject, N))
# #SAVE roilist, nvox
# ROILIST = [r for r in topN]
# ROILIST.append(np.sum(mask))
# ROILIST = pd.DataFrame(ROILIST)
# ROILIST.to_csv("./{}/{}/output/top{}.csv".format(roiloc, subject, N))





def plot():
    # code to load and compare the result of above:
    # SummaryAccuracy.py
    from glob import glob
    import numpy as np
    di="/gpfs/milgram/project/turk-browne/jukebox/ntb/projects/sketchloop02/subjects/"

    subs=glob(f"{di}[0,1]*_neurosketch")
    subs=[sub.split("/")[-1].split("_")[0] for sub in subs]
    subjects=""
    for sub in subs:
        subjects=subjects+sub+" "
        
        
    testDir='/gpfs/milgram/project/turk-browne/projects/rtTest/'
    subjects=subs #["0110171", "0110172", "0111171"]
    hemis=["lh", "rh"]

    wangAcc=np.zeros((50,len(subs)))
    roiloc="wang2014"
    for sub_i,sub in enumerate(subjects):
        for num in range(1,51):
            try:
                wangAcc[num-1,sub_i]=np.load(f"{testDir}{roiloc}/{sub}/output/top{num}.npy")
    #             print(f"{roiloc} {sub} {num} ROIs acc={wangAcc[num-1,sub_i]}")
            except:
                pass

    schaeferAcc=np.zeros((300,3))
    roiloc="schaefer2018"
    for sub_i,sub in enumerate(subjects):
        for num in range(1,301):
            try:
                schaeferAcc[num-1,sub_i]=np.load(f"{testDir}{roiloc}/{sub}/output/top{num}.npy")
    #             print(f"{roiloc} {sub} {num} ROIs acc={schaeferAcc[num-1,sub_i]}")
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
    plt.savefig('SummaryAccuracy.png')
    # next step is to use averageAggregatee.sh to cnvert things to standard space and add things together to visualize things.



# subject = "0111171"
# N = 38

# sbatch aggregate.sh 0119171 neurosketch schaefer2018 40 missing part of brain
# sbatch aggregate.sh 0115174 neurosketch schaefer2018 40
# sbatch aggregate.sh 0119172 neurosketch schaefer2018 40
# sbatch aggregate.sh 0119174 neurosketch schaefer2018 40
# sbatch aggregate.sh 0120171 neurosketch schaefer2018 40 missing part of brain
# sbatch aggregate.sh 0112172 neurosketch schaefer2018 40
# sbatch aggregate.sh 0112173 neurosketch schaefer2018 40
# sbatch aggregate.sh 1206162 neurosketch schaefer2018 40
# sbatch aggregate.sh 0112171 neurosketch schaefer2018 40
# sbatch aggregate.sh 0125171 neurosketch schaefer2018 40
# sbatch aggregate.sh 0120173 neurosketch schaefer2018 40



# not exist
# 0112174