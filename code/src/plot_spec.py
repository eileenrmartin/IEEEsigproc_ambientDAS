
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
from reader import readTrace
import fileSet as fs
import regularFileSet as rfs
import os



# info 
nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240

#sys.path.append('/home/ermartin/PassiveSeismicArray')
# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[3]
startParams = int(sys.argv[4])
lastParams = int(sys.argv[5])

# get the list of files where the spectra are written
p = lastParams
sys.path.append(paramsPath+str(p))
from params import *
outFile = open(outfileListFile+'_spec.txt','r')
filenamesBasic = outFile.readlines()
filenamesBasic = [f.strip('\n') for f in filenamesBasic]
filenames = [f+'.npz' for f in filenamesBasic]

# collect all the spectra in all these files
listOfSpecs = []
nWindows = 0
nFrqs = 0
listOfNWindows = []
for f in filenames:
    thisSpec = np.load(f)['arr_0']
    listOfNWindows.append(thisSpec.shape[0])
    nFrqs = thisSpec.shape[1]
    listOfSpecs.append(thisSpec)
    nWindows = nWindows+listOfNWindows[-1]
allSpec = np.zeros((nWindows,nFrqs),dtype=np.float32)
windowIdx = 0
for i,f in enumerate(filenames):
    allSpec[windowIdx:windowIdx+listOfNWindows[i],:] = listOfSpecs[i]
    windowIdx = windowIdx + listOfNWindows[i]

secondsPerDay = float(24*3600)
nDays = windowIdx*secondsPerWindowOffset/secondsPerDay
epsilon = 0.0001
allSpec = allSpec + epsilon
maxVal = np.percentile(np.log(allSpec),99)
minVal = np.percentile(np.log(allSpec),5)
plt.imshow(np.log(np.transpose(allSpec)),interpolation='nearest',aspect='auto',cmap=plt.get_cmap('inferno'),vmax=maxVal,vmin=minVal,extent=[0,nDays,maxFrq,maxFrq/float(nFrqs)])
plt.colorbar()
plt.title('strain rate spectrum')
plt.ylabel('frequency (Hz)')
plt.xlabel('days after Sep. 10, 2016')
plt.savefig(outfilePath+'spec_plot.pdf')
plt.clf()


