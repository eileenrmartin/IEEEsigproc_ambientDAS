import numpy as np
import matplotlib.pyplot as plt
import sys
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *

srcFile = open(srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
srcChannels = [int(ch) for ch in srcChannelsStrings]

# get the list of all the files
outFileList = open(outfileListFile,'r')
listOfAllFiles = outFileList.readlines()
listOfAllFiles = [(f.strip('\n'))+'.npz' for f in listOfAllFiles]
outFileList.close()

for ch in srcChannels:
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
    secondsToPlot = 1.0
    samplesPerSecond = 50
    nSampsToPlot = int(samplesPerSecond*secondsToPlot)
    xCorr = xCorr[:,:nSampsToPlot]
    

    # plot that average
    clipVal = np.percentile(xCorr,99.8)
    plt.imshow(xCorr,aspect='auto',interpolation='nearest',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[0,secondsToPlot,endCh,startCh])
    plt.ylabel('channel (8 m/channel)')
    plt.xlabel('time lag (sec)')
    plt.title('one bit cross correlation, virtual source '+str(ch))
    plt.colorbar()
    plt.show()
    plt.clf()

    # plot some nearby wiggles
    #nChannelsPlottedPerSide = 25
    #for recCh in range(ch-nChannelsPlottedPerSide,ch+nChannelsPlottedPerSide):
    #    color = float(recCh-(ch-nChannelsPlottedPerSide))/float(2*nChannelsPlottedPerSide)
    #    plt.plot(10*(recCh-(ch-nChannelsPlottedPerSide))+xCorr[recCh-startCh,:],c=(0,color,1.0-color),linewidth=1)
    #plt.show()
    #plt.clf()
