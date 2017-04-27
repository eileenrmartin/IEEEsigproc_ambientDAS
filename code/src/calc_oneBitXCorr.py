#cimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
import xcorr

sys.path.append('/home/ermartin/PassiveSeismicArray')
# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *
import fileSet as fs
import regularFileSet as rfs

nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240
def readTrace(infile,nSamples,dataLen,traceNumber,endian,startSample,nSamplesToRead):
    '''infile is .sgy, nSamples is the number of samples per sensor, and traceNumber is the sensor number (start with 1),dataLen is number of bytes per data sample'''

    fin = open(infile, 'rb') # open file for reading binary mode
    startData = nTxtFileHeader+nBinFileHeader+nTraceHeader+(traceNumber-1)*(nTraceHeader+dataLen*nSamples)+startSample*dataLen
    fin.seek(startData)
    thisTrace = np.zeros(nSamplesToRead)
    for i in range(nSamplesToRead):
       	# was >f before
       	thisTrace[i] = struct.unpack(endian+'f',fin.read(dataLen))[0]
    fin.close()
    return thisTrace


#cdef extern from "xcorrmodule.h":
#	cdef cppclass 
        
def crossCorrOneBit(virtualSrcTrace, allOtherReceiversTraces, nLagSamples): #, version = 'C'):
    #if(version == 'C'):
	#xCorr = xcorr.oneBitXCorr(virtualSrcTrace, allOtherReceiversTraces, nLagSamples)
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
startYr = startTime.year
startMo = startTime.month
startDay = startTime.day
startHr = startTime.hour
startMin = startTime.minute
startSec = startTime.second
startMil = int(0.001*startTime.microsecond)
regFileSet = rfs.regularFileSet(parts,startYr,startMo,startDay,startHr,startMin,startSec,startMil,secondsPerFile,nFiles)
endTime = startTime + dt.timedelta(seconds=secondsPerFile*nFiles-1) # last time in last file
fileList = regFileSet.getFileNamesInRange(startTime,endTime)
# read a little header info about sample rate and number of channels
import os
nBytesPerFile = os.path.getsize(fileList[0])
nChannelsTotal = 620
bytesPerChannel = (nBytesPerFile-nTxtFileHeader-nBinFileHeader-240*nChannelsTotal)/nChannelsTotal
samplesPerChannel = bytesPerChannel/4
samplesPerSecond = samplesPerChannel/secondsPerFile 
print("samples per second "+str(samplesPerSecond))
NyquistFrq = float(samplesPerSecond)/2.0
samplesPerFile = samplesPerSecond*secondsPerFile

# figure out each window size
samplesPerWindow = secondsPerWindowWidth*samplesPerSecond
windowOffset = dt.timedelta(seconds=secondsPerWindowOffset)
windowLength = dt.timedelta(seconds=secondsPerWindowWidth)
currentWindowStartTime = startTime
currentWindowEndTime = currentWindowStartTime + windowLength

outfileList = []

# for each time window, do the cross correlation and write its xcorr to a file
while currentWindowEndTime < endTime:
    print(currentWindowStartTime)

    thisWindowsFileSet= regFileSet.getFileNamesInRange(currentWindowStartTime,currentWindowEndTime)

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
    dataRate[dataRate >= 0] = 1
    dataRate[dataRate <= 0] = -1
    dataRate = dataRate.astype(np.int32)

    # Fantine will do some muting in here *******

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
