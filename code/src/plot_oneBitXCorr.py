import numpy as np
import matplotlib.pyplot as plt
import sys
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *
import datetime as dt


filteredFlag = sys.argv[2]

# read the source channels
srcFile = open(srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
srcChannels = [int(ch) for ch in srcChannelsStrings]

# get the list of all the files
outFileList = open(outfileListFile+'_'+filteredFlag+'.txt','r')
listOfAllFilesBasic = outFileList.readlines()
listOfAllDataFiles = [(f.strip('\n'))+'_data.npz' for f in listOfAllFilesBasic]
listOfAllHeaderFiles = [(f.strip('\n'))+'_headers.txt' for f in listOfAllFilesBasic]
outFileList.close()

# intervals over which convergences against the whole day will be calculated *** need to add this for convergence tests****
intervalsHrs = [1,2,4,6,8]

myXCorrList = []
listOfHrs = []
totalNHrs = 0

for i,filename in enumerate(listOfAllDataFiles):
	# load data
	xCorr = np.load(filename)['arr_0']
	xCorr = xCorr.astype(np.float32)
	myXCorrList.append(xCorr)
	print(xCorr.shape)
	# get dimensions
	nHrs = xCorr.shape[0]
	totalNHrs = nHrs + totalNHrs
	nSrcCh = xCorr.shape[1]
	nRecCh = xCorr.shape[2]
	nLagIdx = xCorr.shape[3]
	
	# ****read header info from corresponding header file*****
	minLag = -xCorrMaxTimeLagSeconds
	maxLag = xCorrMaxTimeLagSeconds
	hdrFile = open(listOfAllHeaderFiles[i],'r')
	hdrLines = hdrFile.readlines()
	thirdLineList = hdrLines[2].split()
	dateStr = thirdLineList[-2]
	dateList = dateStr.split('-')
	timeStr = thirdLineList[-1]
	timeList = timeStr.split(':')
	startHr =  dt.datetime(int(dateList[0]),int(dateList[1]),int(dateList[2]),int(timeList[0]))
	
	for h in range(nHrs):
		listOfHrs.append(startHr + dt.timedelta(seconds=3600*h))

# put all the cross correlations into one array
xCorr = np.empty((totalNHrs,nSrcCh,nRecCh,nLagIdx),dtype=np.float32)
currentHr = 0
for subXCorr in myXCorrList:
	xCorr[currentHr:currentHr+subXCorr.shape[0],:,:,:] = subXCorr

# calculate the average correlation over the day and plot for each source channel
avgCorr = np.sum(xCorr,axis=0)/totalNHrs
for ich,ch in enumerate(srcChannels):
	# plot avg corr for this source channel
	avgCorrNormalized = avgCorr[ich,:,:] # normalize channel-wise
	for i in range(startCh,endCh):
		avgCorrNormalized[i-startCh,:] = avgCorrNormalized[i-startCh,:]/np.sum(np.absolute(avgCorrNormalized[i-startCh,:]))
	clipVal = np.percentile(np.absolute(avgCorrNormalized),99)
	plt.imshow(avgCorr[ich,:,:],aspect='auto',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[minLag,maxLag,endCh,startCh])
	plt.ylabel('channel (8 m/channel)')
	plt.xlabel('time lag (seconds)')
	plt.colorbar()
	plt.title('one-bit '+filteredFlag+' avg. over '+str(nHrs)+', virtual source ch. '+str(ch))
	filename = outfilePath+'oneBit_'+filteredFlag+'_avg_vs_'+str(ch)+'_'+str(nHrs)+'hrs_starting_'+str(listOfHrs[0])+'.pdf'
	plt.savefig(filename)
	plt.clf()
	# plot the corr for each hour
	for hr in range(totalNHrs):
		thisHr = listOfHrs[hr]
		clipVal = np.percentile(np.absolute(xCorr[hr,ich,:,:]),99)
		plt.imshow(xCorr[hr,ich,:,:],aspect='auto',interpolation='nearest',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[minLag,maxLag,endCh,startCh])
		plt.ylabel('channel (8 m/channel)')
		plt.xlabel('time lag (seconds)')
		plt.colorbar()
		plt.title('one-bit '+filteredFlag+' at '+str(thisHr)+', virtual source ch. '+str(ch))
		filename = outfilePath+'oneBit_'+filteredFlag+'_at_'+str(thisHr)+'_vs_'+str(ch)+'.pdf'
		plt.savefig(filename)
		plt.clf()
		
			

#for ch in srcChannels:
    ## get the list of file names for this source channel
    #containsPhrase = '_srcCh_'+str(ch)+'_'
    #mySubsetOfFiles = []
    #for fileName in listOfAllFiles:
    #    if containsPhrase in fileName:
    #        mySubsetOfFiles.append(fileName)
    #        listOfAllFiles.remove(fileName)

    # read the cross correlations and average them
    #xCorr = np.load(mySubsetOfFiles[0])['arr_0']
    #xCorr = xCorr.astype(np.float32)
    #nt = xCorr.shape[1]
    #for i,fileName in enumerate(mySubsetOfFiles):
    #    if i>0:
    #        tempCorr = np.load(fileName)['arr_0']
    #       xCorr = xCorr +  tempCorr.astype(np.float32)
    #xCorr = xCorr/len(mySubsetOfFiles)
    
    # symmetrize
    #xCorr = 0.5*(xCorr[:,nt/2:] + np.fliplr(xCorr[:,:nt/2+1]))
    # normalize channel-wise
    #for i in range(startCh,endCh):
    #    xCorr[i-startCh,:] = xCorr[i-startCh,:]/np.sum(np.absolute(xCorr[i-startCh,:]))

    # *******NEEDS TO BE MADE MORE GENERAL******
    #secondsToPlot = 1.0
    #samplesPerSecond = 50
    #nSampsToPlot = int(samplesPerSecond*secondsToPlot)
    #xCorr = xCorr[:,:nSampsToPlot]
    
    # plot that average
    #clipVal = np.percentile(xCorr,99.8)
    #plt.imshow(xCorr,aspect='auto',interpolation='nearest',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[0,secondsToPlot,endCh,startCh])
    #plt.ylabel('channel (8 m/channel)')
    #plt.xlabel('time lag (sec)')
    #plt.title('one bit cross correlation, virtual source '+str(ch))
    #plt.colorbar()
    #plt.show()
    #plt.clf()

    # plot some nearby wiggles
    #nChannelsPlottedPerSide = 25
    #for recCh in range(ch-nChannelsPlottedPerSide,ch+nChannelsPlottedPerSide):
    #    color = float(recCh-(ch-nChannelsPlottedPerSide))/float(2*nChannelsPlottedPerSide)
    #    plt.plot(10*(recCh-(ch-nChannelsPlottedPerSide))+xCorr[recCh-startCh,:],c=(0,color,1.0-color),linewidth=1)
    #plt.show()
    #plt.clf()
