
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
import numpy.fft as ft


# info 
nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240

#sys.path.append('/home/ermartin/PassiveSeismicArray')
# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[3]
startParams = int(sys.argv[4])
lastParams = int(sys.argv[5])

sys.path.append(paramsPath+str(startParams))
import params
sys.path.remove(paramsPath+str(startParams))

outfileList = []


for p in range(startParams,lastParams+1):
    # get all the job info for this subset
    sys.path.append(paramsPath+str(p))
    reload(params)


    # naming convention for input files
    # if working directly on cees-mazama, cees-tool-7/8 use '/data/biondo/DAS/' as first entry of parts
    # if working with '/data/biondo/DAS/' mounted to '/data/', just use '/data/' as first entry of parts
    parts=['/data/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']

    # overwrite with params from command lien
    startCh = int(sys.argv[1])
    endCh = int(sys.argv[2])
    nChannels = endCh-startCh+1

    # starting time and file set organization
    regFileSets = []
    fileList = []
    for idx,startTime in enumerate(params.startTimes):
        nFiles = params.nFiless[idx]
        startMil = int(0.001*startTime.microsecond)
        regFileSets.append(rfs.regularFileSet(parts,startTime.year,startTime.month,startTime.day,startTime.hour,startTime.minute,startTime.second,startMil,params.secondsPerFile,nFiles))

        endTime = startTime + dt.timedelta(seconds=params.secondsPerFile*nFiles-1) # last time in last file
        tempFileList = regFileSets[-1].getFileNamesInRange(startTime,endTime)
        for f in tempFileList:
	    fileList.append(f)
    # read a little header info about sample rate and number of channels
    nBytesPerFile = os.path.getsize(fileList[0])
    nChannelsTotal = 620
    bytesPerChannel = (nBytesPerFile-nTxtFileHeader-nBinFileHeader-240*nChannelsTotal)/nChannelsTotal
    samplesPerChannel = bytesPerChannel/4
    samplesPerSecond = samplesPerChannel/params.secondsPerFile 
    NyquistFrq = float(samplesPerSecond)/2.0
    samplesPerFile = samplesPerSecond*params.secondsPerFile

    # figure out each window size
    samplesPerWindow = params.secondsPerWindowWidth*samplesPerSecond
    windowOffset = dt.timedelta(seconds=params.secondsPerWindowOffset)
    windowLength = dt.timedelta(seconds=params.secondsPerWindowWidth)
    currentWindowStartTime = params.startTimes[0]
    currentWindowEndTime = currentWindowStartTime + windowLength


    regFileSetIdx = 0 # which regular file set in the list are you using now?
    windowIdx = 0 # which index within the spectrum matrix are you at?

    # create the spectrum matrix for the first file set, will append to this later as you go to the next file 
    nUpcomingWindows = params.nFiless[regFileSetIdx]*params.secondsPerFile/params.secondsPerWindowOffset 
    nFrqs = 1+samplesPerWindow/2
    thisSpec = np.zeros((nUpcomingWindows,nFrqs),dtype=np.float32)

    # for each time window, do the cross correlation and write its xcorr to a file
    while currentWindowEndTime < endTime:
        # figure out start time in case there was a jump
        thisWindowsFileSet = regFileSets[regFileSetIdx].getFileNamesInRange(currentWindowStartTime,currentWindowEndTime)
        print(currentWindowStartTime)

        data = np.zeros((nChannels,samplesPerWindow),dtype=np.float32)
        startIdx = 0
        for filename in thisWindowsFileSet: 
            thisFileStartTime = regFileSets[regFileSetIdx].getTimeFromFilename(filename)
            thisFileEndTime = thisFileStartTime + dt.timedelta(seconds=params.secondsPerFile)
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
            thisTrace.filter('bandpass',freqmin=params.minFrq,freqmax=params.maxFrq,corners=4,zerophase=True)
            dataRate[ch-startCh,:] = thisTrace.data

        # get rid of laser drift
        dataRate = dataRate - np.median(dataRate,axis=0)

        # get the spec for this window and append to an array
        fourierTrans = ft.fft(dataRate,axis=1) # take FT spectrum for each channel
	thisSpec[windowIdx,:] = np.sum(np.absolute(fourierTrans[:,:nFrqs]),axis=0)/nChannels # absolute value then average over channel set

        # move on to the next time step
        lastWindowInFileSet = (thisWindowsFileSet[-1] == regFileSets[regFileSetIdx].nameOfLastFile())
        if lastWindowInFileSet: # if moving on to the next file set, figure out how much zero padding is needed
            if(regFileSetIdx < len(regFileSets)-1):
    	        regFileSetIdx = regFileSetIdx + 1
                nextWindowStartTime = params.startTimes[regFileSetIdx]
                nZeroWindows = -1+int((nextWindowStartTime-currentWindowStartTime).total_seconds()/params.secondsPerWindowOffset)
                nUpcomingWindows = params.nFiless[regFileSetIdx]*params.secondsPerFile/params.secondsPerWindowOffset #slight overpadding
                thisSpec = np.pad(thisSpec,((0,nZeroWindows+nUpcomingWindows+1),(0,0)),'constant',constant_values=((0,0),(0,0))) # pad array with zeros for those windows and upcoming ones for next file set
                windowIdx = windowIdx + nZeroWindows
                currentWindowStartTime = nextWindowStartTime
            else:
                break
        else: # just keep marching within this file set
            currentWindowStartTime = currentWindowStartTime + windowOffset
            windowIdx = windowIdx+1
        currentWindowEndTime = currentWindowStartTime + windowLength

    # do the output for this subset of data
    outfileName = params.outfilePath + 'spec'+str(p)+'_chs_'+str(startCh)+'_to_'+str(endCh)
    np.savez(outfileName,thisSpec[:windowIdx+1,:])
    outfileList.append(outfileName)


    sys.path.remove(paramsPath+str(p))


# write the list of output file names
outFile = open(params.outfileListFile+'_spec.txt','w')
for filename in outfileList:
    outFile.write(filename+'\n')
outFile.close()
