
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
from reader import readTrace
#import corrs
import cwt

#sys.path.append('/home/ermartin/PassiveSeismicArray')
# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *
import fileSet as fs
import regularFileSet as rfs

nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240

#cdef extern from "xcorrmodule.h":
#	cdef cppclass 
        
def crossCorrOneBit(virtualSrcTrace, allOtherReceiversTraces, nLagSamples, version = 'python'): #'C'):
    '''version = C or python where that determines which language is used for correlation calculation'''
    #if(version == 'C'):
    #xCorr = corrs*********
    #xCorr = corrs.oneBitXCorr(virtualSrcTrace, allOtherReceiversTraces, nLagSamples)
    #else:
    numberChannels = allOtherReceiversTraces.shape[0]
    nt = allOtherReceiversTraces.shape[1]
    xCorr = np.empty((numberChannels,1+2*nLagSamples),dtype=np.int32)
    sumWidth = nt-2*nLagSamples
    for i in range(-1*nLagSamples,nLagSamples+1):
        startSample = i + nLagSamples
        endSample = startSample + sumWidth
        tempArray = virtualSrcTrace[startSample:endSample]*allOtherReceiversTraces[:,nLagSamples:nLagSamples+sumWidth]
        xCorr[:,i+nLagSamples] = np.sum(tempArray,axis=1)
    return xCorr


# naming convention for input files
parts=['/data/biondo/DAS/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']

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
    startYr = startTime.year
    startMo = startTime.month
    startDay = startTime.day
    startHr = startTime.hour
    startMin = startTime.minute
    startSec = startTime.second
    startMil = int(0.001*startTime.microsecond)
    regFileSets.append(rfs.regularFileSet(parts,startYr,startMo,startDay,startHr,startMin,startSec,startMil,secondsPerFile,nFiles))
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

# load the estimator 
from sklearn.externals import joblib
clusterFileName = 'kmeans.pkl' # or 'aggloCluster.pkl'
estimator = joblib.load(clusterFileName) # ******************************
# **********uncomment estimator = .... as soon as that file is added to git repo ******

# for each time window, do the cross correlation and write its xcorr to a file
while currentWindowEndTime < endTime:

    thisWindowsFileSet = []
    for i,regFileSet in enumerate(regFileSets):
        thisWindowsFileSet= regFileSet.getFileNamesInRange(currentWindowStartTime,currentWindowEndTime)
        if(len(thisWindowsFileSet) > 0):
            currentWindowStartTime = regFileSet.getTimeFromFilename(thisWindowsFileSet[0]) 
            currentWindowEndTime = currentWindowStartTime + windowLength
            break
    print(currentWindowStartTime)


    data = np.zeros((nChannels,samplesPerWindow))
    startIdx = 0
    for filename in thisWindowsFileSet: 
        thisFileStartTime = regFileSet.getTimeFromFilename(filename)
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

    # get rid of laser drift, then do one bit simplification
    dataRate = dataRate - np.median(dataRate,axis=0)
#     dataRate[dataRate >= 0] = 1
#     dataRate[dataRate <= 0] = -1
#     dataRate = dataRate.astype(np.int32)

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
    cwtScales[:,samplesPerWindow,:] = cwtScales[:,(samplesPerWindow - 1),:]

    # figure out which cluster the sample belongs to
    samplingRate = 25
    nSubSamples = int(samplesPerWindow / samplingRate)
    features = np.mean(np.reshape(np.abs(cwtScales), (nChannels, nSubSamples, samplingRate, nf * 2)), axis=2)
    labels = estimator.predict(scale(np.reshape(features, (nChannels * nSubSamples, -1)), copy=False))
    labels = np.reshape(labels, (nChannels, -1))
    
    # mute out the coefficients 
    # TODO : check which cluster number corresponds to the cars after training

    # apply inverse cwt to reconstruct the signal
    for index in range(samplesPerWindow):
        dataRate[:,index] = cwt.icwt(cwtScales[:,index,nf:].T, delta, scales, wf, w0)

    # do the cross-correlations and save them
    nLagSamples = int(xCorrMaxTimeLagSeconds*samplesPerSecond)
    for ch in srcChannels:
        virtualSrcTrace = dataRate[ch-startCh,:]
        xcorr =  crossCorrOneBit(virtualSrcTrace, dataRate, nLagSamples)
        # write the output to a file
        outfileName = outfilePath + 'xcorr_srcCh_'+str(ch)+'_starting_'+str(currentWindowStartTime)
        outfileName = outfileName.replace(" ","_") # don't have a space in the middle of the name
        np.savez(outfileName,xcorr)
        outfileList.append(outfileName)
   
    # move on to the next time step
    currentWindowStartTime = currentWindowStartTime + windowOffset
    currentWindowEndTime = currentWindowStartTime + windowLength



# write the list of output file names
outFile = open(outfileListFile,'w')
for filename in outfileList:
    outFile.write(filename+'\n')
outFile.close()
