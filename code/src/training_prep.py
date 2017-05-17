import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
from reader import readTrace
import cwt
import time

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

# naming convention for input files
# if working directly on cees-mazama, cees-tool-7/8 use '/data/biondo/DAS/' as first entry of parts
# if working with '/data/biondo/DAS/' mounted to '/data/', just use '/data/' as first entry of parts
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

# Parameters for cwt
count = 0
nSamples = secondsPerWindowOffset * samplesPerSecond
samplingRate = 25
nSubSamples = int(nSamples / samplingRate)
start = 5 * samplesPerSecond
end = start + nSamples
minSpaceFrq = 0.04
maxSpaceFrq = 2
nf = 30
delta = 1.0 / float(samplesPerSecond)  
ftime = np.logspace(np.log10(minFrq), np.log10(maxFrq), nf)
fspace = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), nf)
wf = 'morlet'
w0 = 8
scalesTime = cwt.scales_from_fourier(ftime, wf, w0)
scalesSpace = cwt.scales_from_fourier(fspace, wf, w0)
cwtScales = np.empty((nChannels, samplesPerWindow - 1, nf * 2), dtype=np.float64)
features = np.empty((nChannels, nSubSamples, nf * 2), dtype=np.float64)

# for each time window, compute cwt scales and build training data set
while currentWindowEndTime < endTime:
  
  start_time = time.time()

  thisWindowsFileSet = []
  for i,regFileSet in enumerate(regFileSets):
    thisWindowsFileSet= regFileSet.getFileNamesInRange(currentWindowStartTime,currentWindowEndTime)
    if(len(thisWindowsFileSet) > 0):
      currentWindowStartTime = regFileSet.getTimeFromFilename(thisWindowsFileSet[0]) 
      currentWindowEndTime = currentWindowStartTime + windowLength
      break
  print(currentWindowStartTime)

  data = np.zeros((nChannels,samplesPerWindow), dtype=np.float64)
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

  # get rid of laser drift,
  dataRate = dataRate - np.median(dataRate, axis=0)
  
  # compute cwt over time
  for index, trace in enumerate(dataRate):
    cwtScales[index,:,:nf] = np.abs(cwt.cwt(trace, delta, scalesTime, wf, w0).T)
  # compute cwt over space
  for index, channel in enumerate(dataRate.T):
    cwtScales[:,index,nf:] = np.abs(cwt.cwt(channel, delta, scalesSpace, wf, w0).T)

  # average them over 0.5 time windows and cut off boundaries
  features = np.mean(np.reshape(cwtScales[:,start:end,:], (nChannels, nSubSamples, samplingRate, nf * 2)), axis=2)
  
  # save 
  outfileName = outfilePath + 'cwt_%02d_%03d' % (startTime.day, count)
  np.save(outfileName, features)
  outfileList.append(outfileName)

  count = count + 1
  end_time = time.time()
  print('time elapsed', end_time - start_time)

  # move on to the next time step
  currentWindowStartTime = currentWindowStartTime + windowOffset
  currentWindowEndTime = currentWindowStartTime + windowLength

# write the list of output file names
outFile = open(outfileListFile,'w')
for filename in outfileList:
  outFile.write(filename + '\n')
outFile.close()

