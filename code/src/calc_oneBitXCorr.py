
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
from reader import readTrace
import onebitcrosscorr_module
import cwt

#sys.path.append('/home/ermartin/PassiveSeismicArray')
# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *
import fileSet as fs
import regularFileSet as rfs

filteredFlag = sys.argv[2]
if((filteredFlag != 'baseline') and (filteredFlag != 'filtered')):
	print("ERROR: sys.argv[2] must be baseline or filtered")

nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240

        
def crossCorrOneBit(virtualSrcTrace, allOtherReceiversTraces, nLagSamples, version = 'C'):
    '''version = C or python where that determines which language is used for correlation calculation'''
    numberChannels = allOtherReceiversTraces.shape[0]
    xCorr = np.empty((numberChannels,1+2*nLagSamples),dtype=np.int32)
    if(version == 'C'):
	nSamples = virtualSrcTrace.size
	# next line should overwrite xCorr entries with one bit cross-correlations
    	flag = onebitcrosscorr_module.onebitcrosscorr_func(virtualSrcTrace, nSamples, allOtherReceiversTraces, numberChannels, xCorr, nLagSamples)
    else: # just do the one bit correlation in 
    	nt = allOtherReceiversTraces.shape[1]
    	sumWidth = nt-2*nLagSamples
	# do +- 1 bit thresholding
	vsOneBit = np.ones_like(virtualSrcTrace)
	# ****do threhsolding***
	recsOneBit = np.ones_like(allOtherReceiversTraces)
	# ****do thresholding
    	for i in range(-1*nLagSamples,nLagSamples+1):
    	    startSample = i + nLagSamples
    	    endSample = startSample + sumWidth
    	    tempArray = vsOneBit[startSample:endSample]*recsOneBit[:,nLagSamples:nLagSamples+sumWidth]
    	    xCorr[:,i+nLagSamples] = np.sum(tempArray,axis=1)
    return xCorr


# naming convention for input files
# if working directly on cees-mazama, cees-tool-7/8 use '/data/biondo/DAS/' as first entry of parts
# if working with '/data/biondo/DAS/' mounted to '/data/', just use '/data/' as first entry of parts
parts=['/data/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']

srcFile = open(srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
srcChannels = [int(ch) for ch in srcChannelsStrings] 
nChannels = endCh-startCh+1 # number of receiver channels

# starting time and file set organization
regFileSets = []
fileList = []
for idx,startTime in enumerate(startTimes):
    nFiles = nFiless[idx]
    startMil = int(0.001*startTime.microsecond)
    regFileSets.append(rfs.regularFileSet(parts,startTime.year,startTime.month,startTime.day,startTime.hour,startTime.minute,startTime.second,startMil,secondsPerFile,nFiles))
    endTime = startTime + dt.timedelta(seconds=secondsPerFile*nFiles-1) # last time in last file
    tempFileList = regFileSets[-1].getFileNamesInRange(startTime,endTime)
    for f in tempFileList:
	fileList.append(f)
# read a little header info about sample rate and number of channels
import os
nBytesPerFile = os.path.getsize(fileList[0])
nChannelsTotal = 620
bytesPerChannel = (nBytesPerFile-nTxtFileHeader-nBinFileHeader-240*nChannelsTotal)/nChannelsTotal
samplesPerChannel = bytesPerChannel/4
samplesPerSecond = samplesPerChannel/secondsPerFile 
NyquistFrq = float(samplesPerSecond)/2.0
samplesPerFile = samplesPerSecond*secondsPerFile

# figure out each window size
samplesPerWindow = secondsPerWindowWidth*samplesPerSecond
windowOffset = dt.timedelta(seconds=secondsPerWindowOffset)
windowLength = dt.timedelta(seconds=secondsPerWindowWidth)
currentWindowStartTime = startTimes[0]
currentWindowEndTime = currentWindowStartTime + windowLength

outfileList = []

# set up the output, one cross correlation per virtual source receiver pair per hour
overallStartTimeForThisJob = min(startTimes)
firstHr = dt.datetime(overallStartTimeForThisJob.year, overallStartTimeForThisJob.month, overallStartTimeForThisJob.day, overallStartTimeForThisJob.hour,0,0,0) 
overallEndTimeForThisJob = endTime # assume lists within job parameters were put in in order ***shoudl generalize***
lastIncludedHr = dt.datetime(overallEndTimeForThisJob.year, overallEndTimeForThisJob.month, overallEndTimeForThisJob.day, overallEndTimeForThisJob.hour,0,0,0)
nHrs = int((lastIncludedHr-firstHr).total_seconds())/3600 # number of hours output with correlations
nLagSamples = int(xCorrMaxTimeLagSeconds*samplesPerSecond)
nLagSamplesTotal = 2*nLagSamples+1 # number of cross correlation lags 
outputCorrelations = np.zeros((nHrs,len(srcChannels),nChannels,nLagSamplesTotal),dtype=np.float32)
nWindowsPerCorrelation = np.zeros(nHrs) # each hour will have a counter indicating how many windows contributed to its avg correlation

if(filteredFlag == 'filtered'):
    # load the estimator 
    from sklearn.externals import joblib
    clusterFileName = '/mnt/kmeans.pkl' # or '/mnt/aggloCluster.pkl' # this is actually in /scratch/fantine/das on cees but I bind that to /mnt
    estimator = joblib.load(clusterFileName) 


regFileSetIdx = 0 # which regular file set in the list are you using now?
# for each time window, do the cross correlation and write its xcorr to a file
while currentWindowEndTime < endTime:

    # figure out start time in case there was a jump
    thisWindowsFileSet = regFileSets[regFileSetIdx].getFileNamesInRange(currentWindowStartTime,currentWindowEndTime)
    print(currentWindowStartTime)

    # figure out which hour this window is assigned to
    thisHrIdx = int((currentWindowStartTime-overallStartTimeForThisJob).total_seconds()/3600)
    nWindowsPerCorrelation[thisHrIdx] = nWindowsPerCorrelation[thisHrIdx] + 1

    data = np.zeros((nChannels,samplesPerWindow),dtype=np.float32)
    startIdx = 0
    for filename in thisWindowsFileSet: 
        thisFileStartTime = regFileSets[regFileSetIdx].getTimeFromFilename(filename)
        thisFileEndTime = thisFileStartTime + dt.timedelta(seconds=secondsPerFile)
        startIdxReading = 0 # start index to read in filename
        if currentWindowStartTime > thisFileStartTime:
            secondsAfterStart = (currentWindowStartTime - thisFileStartTime).total_seconds()
            startIdxReading = int(samplesPerSecond*secondsAfterStart)
        endIdxReading = samplesPerFile
        if currentWindowEndTime < thisFileEndTime:
            secondsAfterStart = (currentWindowEndTime - thisFileStartTime).total_seconds()
            endIdxReading = int(samplesPerSecond*secondsAfterStart)
        nIdxToRead = endIdxReading-startIdxReading
        endIdx = startIdx + nIdxToRead # end of this subset of data in array

        # read the data
        for ch in range(startCh,endCh+1):
            data[ch-startCh,startIdx:endIdx] =  readTrace(filename,samplesPerFile,4,ch,'>',startIdxReading,nIdxToRead)
        startIdx = endIdx # index in data array
  
        
    # take time derivatives
    dataRate = data[:,1:] - data[:,:-1]


    
    # do bandpass from minFrq to maxFrq
    for ch in range(startCh,endCh+1):
        thisTrace = obspy.core.trace.Trace(data=dataRate[ch-startCh,:],header={'delta':1.0/float(samplesPerSecond),'sampling_rate':samplesPerSecond})
        thisTrace.filter('bandpass',freqmin=minFrq,freqmax=maxFrq,corners=4,zerophase=True)
        dataRate[ch-startCh,:] = thisTrace.data

    # get rid of laser drift,
    dataRate = dataRate - np.median(dataRate,axis=0)

    # get rid of cars if you're filtering them out
    if(filteredFlag == 'filtered'):
        # compute cwt scales
        nf = 25
        delta = 1.0 / float(samplesPerSecond)
        # compute cwt over time
        f = np.logspace(np.log10(minFrq), np.log10(maxFrq), nf)
        wf = 'morlet'
        w0 = 8
        scales = cwt.scales_from_fourier(f, wf, w0)
        cwtScales = np.empty((nChannels, samplesPerWindow, nf * 2), dtype='complex128')
        for index, trace in enumerate(dataRate):
            cwtScales[index,:(samplesPerWindow - 1),:nf] = cwt.cwt(trace, delta, scales, wf, w0).T
        # compute cwt over space
        minSpaceFrq = 0.5
        maxSpaceFrq = 50 
        f = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), nf)
        scales = cwt.scales_from_fourier(f, wf, w0)
        for index, channel in enumerate(dataRate.T):
            cwtScales[:,index,nf:] = cwt.cwt(channel, delta, scales, wf, w0).T

        # padding to make my life easier
        np.pad(cwtScales,((0,0),(0,1),(0,0)),'edge') # just add padding to the middle axis, and edge value fill in should perform the operation on the next line
        #cwtScales[:,samplesPerWindow,:] = cwtScales[:,(samplesPerWindow - 1),:]

        # figure out which cluster the sample belongs to
        samplingRate = 25
        nSubSamples = int(samplesPerWindow / samplingRate)
        features = np.mean(np.reshape(np.abs(cwtScales), (nChannels, nSubSamples, samplingRate, nf * 2)), axis=2)
        labels = estimator.predict(scale(np.reshape(features, (nChannels * nSubSamples, -1)), copy=False))
        labels = np.reshape(labels, (nChannels, -1))
    
        # mute out the coefficients 
        # TODO : check which cluster number corresponds to the cars after training

        # apply inverse cwt to reconstruct the signal
        for index in range(samplesPerWindow-1):  # **** changed this to samplesPerWindow-1 since dataRate is 1 shorter than data
            dataRate[:,index] = cwt.icwt(cwtScales[:,index,nf:].T, delta, scales, wf, w0)

    # do the cross-correlations and save them
    for idxchannel,ch in enumerate(srcChannels):
        virtualSrcTrace = dataRate[ch-startCh,:]
        xcorr =  crossCorrOneBit(virtualSrcTrace, dataRate, nLagSamples, 'C') # do the cross correlation
	outputCorrelations[thisHrIdx,idxchannel,:,:] = outputCorrelations[thisHrIdx,idxchannel,:,:] + xcorr.astype(np.float32)



    # move on to the next time step
    lastWindowInFileSet = (thisWindowsFileSet[-1] == regFileSets[regFileSetIdx].nameOfLastFile())
    if lastWindowInFileSet: # if moving on to the next file set
	regFileSetIdx = regFileSetIdx + 1
        currentWindowStartTime = startTimes[regFileSetIdx]
    else: # if staying within the same file set, just march along
        currentWindowStartTime = currentWindowStartTime + windowOffset
    currentWindowEndTime = currentWindowStartTime + windowLength




# normalize/average each hour's correlations by the number of windows contributing to it
for hr in range(nHrs):
    if(nWindowsPerCorrelation[hr] > 0): 
        outputCorrelations[hr,:,:,:] = outputCorrelations[hr,:,:,:] / float(nWindowsPerCorrelation[hr])

###### output ######
outfileName = outfilePath + 'xcorr_starting_'+str(firstHr)+'_'+filteredFlag
outfileName = outfileName.replace(" ","_")
outfileList.append(outfileName)

# write the output correlation metadata  # #*********shoudl move this over to reader.py*****
outfileHeaderName = outfileName+'_headers.txt'
outfile = open(outfileHeaderName,'w')
outfile.write('dimensions (slow,mid,mid,fast) : hour \t source channel (index within selected source channels) \t receiver channel (index within selected receiver channels) \t time lag of correlation \n')
outfile.write('dimensions (slow,mid,mid,fast) : '+str(nHrs)+'\t'+str(len(srcChannels))+'\t'+str(nChannels)+'\t'+str(nLagSamplesTotal)+'\n')
outfile.write('first hour starts at time : '+str(firstHr)+'\n')
outfile.write('number of hours : '+str(nHrs)+'\n')
outfile.write('number of virtual source channels: '+str(len(srcChannels))+'\n')
outfile.write('virtual source channels, indices within full data set : \n')
for ch in srcChannels:
    outfile.write(str(ch)+'\n')
outfile.write('number of receiver channels : '+str(nChannels)+'\n')
outfile.write('minimum receiver channel : '+str(startCh)+'\n')
outfile.write('maximum receiver channel : '+str(endCh)+'\n')
outfile.write('number of time lags total (including -,0,+) : '+str(nLagSamplesTotal)+'\n')
outfile.write('minimum correlation time lag (seconds): '+str(-xCorrMaxTimeLagSeconds)+'\n')
outfile.write('maximum correlation time lag (seconds): '+str(xCorrMaxTimeLagSeconds)+'\n')
outfile.close()

# write the output correlation data
outfileDataName = outfileName+'_data' # will end in .npz
np.savez(outfileDataName,outputCorrelations)

# write the list of output file names
outFile = open(outfileListFile+'_'+filteredFlag+'.txt','w')
for filename in outfileList:
    outFile.write(filename+'\n')
outFile.close()
