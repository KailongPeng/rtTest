import os
print(f"conda env={os.environ['CONDA_DEFAULT_ENV']}") 
import sys,pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import nibabel as nib

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def Class(data, bcvar):
    metas = bcvar[0]
    data4d = data[0]
    print(data4d.shape)

    accs = []
    for run in range(6):
        print(f"run={run}")
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

def getMask(topN, subject):
    workingDir="/gpfs/milgram/project/turk-browne/projects/rtTest/"
    for pn, parc in enumerate(topN):
        _mask = nib.load(workingDir+"/{}/{}/{}".format(roiloc, subject, parc))
        aff = _mask.affine
        _mask = _mask.get_data()
        _mask = _mask.astype(int)
        # say some things about the mask.
        mask = _mask if pn == 0 else mask + _mask
        mask[mask>0] = 1
    return mask

tmpFile = sys.argv[1]
print(f"tmpFile={tmpFile}")
[_topN,subject,dataSource,roiloc,N] = load_obj(tmpFile)
[bcvar,runs] = load_obj(f"./tmp_folder/{subject}_{dataSource}_{roiloc}_{N}") 
_mask=getMask(_topN,subject) ; print('mask dimensions: {}'. format(_mask.shape)) ; print('number of voxels in mask: {}'.format(np.sum(_mask)))
_runs = [runs[:,:,_mask==1]] ; print("Runs shape", _runs[0].shape)

# [_runs,bcvar] = load_obj(tmpFile)
sl_result = Class(_runs, bcvar)

np.save(tmpFile+'_result',sl_result)

print(f"sl_result={sl_result}")