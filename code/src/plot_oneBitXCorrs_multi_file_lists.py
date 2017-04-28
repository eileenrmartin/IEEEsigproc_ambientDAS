import numpy as np
import matplotlib.pyplot as plt
import sys
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *

filenameStart = sys.argv[2]

srcFile = open(srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
srcChannels = [int(ch) for ch in srcChannelsStrings]

# get the list of all the files
listOfAllFiles = []
for outfileListFile in outfileListFiles:
    outFileList = open(outfileListFile,'r')
    sublistOfAllFiles = outFileList.readlines()
    sublistOfAllFiles = [(f.strip('\n'))+'.npz' for f in sublistOfAllFiles]
    for s in sublistOfAllFiles:
        listOfAllFiles.append(s)
    outFileList.close()

# read all the receiver locations
recXs = np.zeros(endCh-startCh+1)
recYs = np.zeros(endCh-startCh+1)
recLocFile = open(recLocationsFilename,'r')
firstChannel = True
ch = startCh
for line in recLocFile:
    line = line.strip('\n')
    stringList = line.split()
    ch = int(stringList[0])
    if (ch>=startCh) and (ch<=endCh):
        recXs[ch-startCh] = float(stringList[1])
        recYs[ch-startCh] = float(stringList[2])
        # if it's the first channel to have the location read, have all earlier channels in same spot
        if firstChannel:
            for earlierChannel in range(startCh,ch):
                recXs[earlierChannel-startCh] = recXs[ch-startCh]
                recYs[earlierChannel-startCh] = recYs[ch-startCh]
            firstChannel = False # no longer the first channel
# after the last channel fill in rest to be same location
while ch <= endCh:
    recXs[ch-startCh] = recXs[ch-startCh-1]
    recYs[ch-startCh] = recYs[ch-startCh-1]
    ch = ch + 1

for ch in srcChannels:
    srcChX = recXs[ch-startCh]
    srcChY = recYs[ch-startCh]

    # get the list of file names for this source channel
    containsPhrase = '_srcCh_'+str(ch)+'_'
    mySubsetOfFiles = []
    for fileName in listOfAllFiles:
        if containsPhrase in fileName:
            mySubsetOfFiles.append(fileName)
            listOfAllFiles.remove(fileName)

    # read the cross correlations and average them
    xCorr = np.load(mySubsetOfFiles[0])['arr_0']
    xCorr = xCorr.astype(np.float32)
    nt = xCorr.shape[1]
    for i,fileName in enumerate(mySubsetOfFiles):
        if i>0:
            tempCorr = np.load(fileName)['arr_0']
            xCorr = xCorr +  tempCorr.astype(np.float32)
    xCorr = xCorr/len(mySubsetOfFiles)
    
    # symmetrize
    xCorr = 0.5*(xCorr[:,nt/2:] + np.fliplr(xCorr[:,:nt/2+1]))
    # normalize channel-wise
    for i in range(startCh,endCh):
        xCorr[i-startCh,:] = xCorr[i-startCh,:]/np.sum(np.absolute(xCorr[i-startCh,:]))

    # *******NEEDS TO BE MADE MORE GENERAL******
    secondsToPlot = 1.5
    samplesPerSecond = 50
    nSampsToPlot = int(samplesPerSecond*secondsToPlot)
    xCorr = xCorr[:,:nSampsToPlot]
    
    # predict arrival times of a 500 m/s wave starting at this virtual source
    distances = np.sqrt(10+(srcChX-recXs)**2 + (srcChY-recYs)**2)
    channels = np.arange(startCh,endCh+1)
    v1 = 400.0 # m/s
    arrivalTimesV1 = distances/v1
    v2 = 1200.0 # m/s
    arrivalTimesV2 = distances/v2

    # plot that average and overlay with any velocity plots
    clipVal = np.percentile(xCorr,99.8)
    plt.imshow(xCorr,aspect='auto',interpolation='nearest',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[0,secondsToPlot,endCh,startCh])
    plt.ylabel('channel (8 m/channel)')
    plt.xlabel('time lag (sec)')
    plt.title('one bit cross-correlation, virtual source '+str(ch))
    plt.colorbar()
    #plt.scatter(arrivalTimesV1[1:-1],channels[1:-1],s=0.5,c='k')
    #plt.scatter(arrivalTimesV2[1:-1],channels[1:-1],s=0.5,c='g')
    plt.savefig('fig/'+filenameStart+'_vs_ch_'+str(ch)+'.pdf')
    plt.clf()

    # plot some nearby wiggles
    #startChPlots = 100
    #endChPlots = 210
    #scale = 30
    #for recCh in range(startChPlots,endChPlots):
    #    color = float(recCh-startChPlots)/float(endChPlots-startChPlots)
    #    plt.plot(np.linspace(0.0,secondsToPlot,nt/2+1),recCh+scale*xCorr[recCh-startCh,:],c=(0,color,1.0-color),linewidth=1)
    #    plt.ylabel('channel (8 m/channel)')
    #    plt.title('one bit cross-correlation, virtual source '+str(ch))
    #plt.savefig('fig/xcorrWiggle_1to12_vs_ch_'+str(ch)+'.png')
    #plt.clf()

    
