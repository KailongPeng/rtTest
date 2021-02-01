# pollDICOM.py
# Realtime functional MRI minimal processing stream.
# Continuously checks /gpfs/milgram/project/realtime/DICOM/{subject} for new files.
# When a new file is detected it sends a dummy message to the Linux computer at BIC
# via TCP/IP socket communication
#
# Richard Watts, August 2019

# Press ctrl-C to stop

import os, glob, time, socket
import nibabel as nib
import dicom
from nibabel.nicom import dicomreaders
import numpy as np

DICOMfolder = '/gpfs/milgram/project/realtime/DICOM' # Actual realtime folder
#DICOMfolder = '/gpfs/milgram/project/bicadmin/rw582/MyProjects/realtime/DICOM' # Dummy folder with R/W access

linuxIP = '172.28.143.32'
linuxPort = 3000
feedbackDataSize = 1024*1024
verbose = True


feedbackData = np.random.bytes(feedbackDataSize)

# Find most recent subject folder based on timestamp
print(DICOMfolder)
print(os.listdir(DICOMfolder))
print(len(os.listdir(DICOMfolder)))
subjectFolders = glob.glob(os.path.join(DICOMfolder,'*')) # * means all if need specific format then *.csv
print(subjectFolders)
latestSubjectFolder = max(subjectFolders, key=os.path.getctime)
print('Most recent folder is {0}'.format(latestSubjectFolder))

# Find most recent DICOM file to determine the new file to expect
DICOMfiles = glob.glob(os.path.join(latestSubjectFolder, '*.dcm'))
if len(DICOMfiles)>0:
    latestDICOMFile = os.path.basename(max(DICOMfiles, key=os.path.getctime))
    print('Most recent DICOM file is {0}'.format(latestDICOMFile))

    filenameParts = latestDICOMFile.split('_')
    currentSubject = int(filenameParts[0])
    latestSeries = int(filenameParts[1])
    latestImage = int(filenameParts[1].split('.')[0])

    print('Most recent subject {0:03d} series {1:06d} image {2:06d}'.format(currentSubject, latestSeries, latestImage))
    currentSeries = latestSeries + 1
    currentImage = 1

else: # No DICOM files in folder
    currentSubject = 1
    currentSeries = 1
    currentImage = 1

# Create the socket to the Linux computer
linuxSocket = socket.socket()
print('Connecting to Linux computer')
linuxSocket.connect((linuxIP, linuxPort))


try:
    while True:
        nextFilename = '{0:03d}_{1:06d}_{2:06d}.dcm'.format(currentSubject, currentSeries, currentImage)
        nextFullFilename = os.path.join(latestSubjectFolder, nextFilename)

        if verbose:
            print('Waiting for {0}'.format(nextFullFilename))

        # Check for new file every 1ms
        while not os.path.exists(nextFullFilename):
            time.sleep(0.001)

        # Maybe useful to have a timestamp
        print(time.time())

        # Give a little time for the file to be fully transferred
        time.sleep(0.01)
        if verbose:
            print('File size is {0} bytes'.format(os.path.getsize(nextFullFilename)))

        # read in, convert to nifti, analyze, send file back

        dicomObject = dicom.read_file(nextFullFilename)
        niftiObject = dicomreaders.mosaic_to_nii(dicomObject)
        dat = niftiObject.get_data()
        print(dat.shape)

        currentImage = currentImage + 1

        if verbose:
            print('Send')
        linuxSocket.send(feedbackData)

except (KeyboardInterrupt):
    print('Close')
    linuxSocket.close()
