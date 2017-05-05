#cimport numpy as np
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import struct
import obspy
import sys
from reader import *
import cwt
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AggloCluster

# get parameters: startTime, secondsPerFile, secondsPerWindowWidth, secondsPerWindowOffset, xCorrMaxTimeLagSeconds, nFiles, outfilePath, outfileList, srcChList, startCh, endCh, minFrq, maxFrq
paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *
import fileSet as fs
import regularFileSet as rfs

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

count = 0 
# these are 4 hour windows, there are 7 * 6 within one week and we downsample them over 0.5 sec (25 samples)
nSamples = secondsPerWindowOffset * samplesPerSecond
samplingRate = 25
nSubSamples = int(nSamples / samplingRate)
nf = 25
trainingData = np.empty((nChannels, nSubSamples * 7 * 6, nf * 2))

# for each time window, compute cwt scales and build training data set
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
  dataRate = dataRate - np.median(dataRate, axis=0)

  # compute cwt scales
  delta = 1.0 / float(samplesPerSecond)
  # compute cwt over time
  f = np.logspace(np.log10(minFrq), np.log10(maxFrq), nf)
  wf = 'morlet'
  w0 = 8
  scales = cwt.scales_from_fourier(f, wf, w0)
  cwtScales = np.empty((nChannels, samplesPerWindow - 1, nf * 2))
  for index, trace in enumerate(dataRate):
    cwtScales[index,:,:nf] = np.abs(cwt.cwt(trace, delta, scales, wf, w0).T)
  # compute cwt over space
  minSpaceFrq = 0.5
  maxSpaceFrq = 50 
  f = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), nf)
  scales = cwt.scales_from_fourier(f, wf, w0)
  for index, channel in enumerate(dataRate.T):
    cwtScales[:,index,nf:] = np.abs(cwt.cwt(channel, delta, scales, wf, w0).T)

  # cut off boundaries 
  start = 60 * samplesPerSecond
  end = start + nSamples
  cwtScales = cwtScales[:,start:end,:]

  # save the cwt scales
  outfileName = outfilePath + 'cwt_' + str(currentWindowStartTime + dt.timedelta(seconds=60))
  outfileName = outfileName.replace(" ","_") # don't have a space in the middle of the name
  np.savez(outfileName, cwtScales)
  outfileList.append(outfileName)

  # subsample to 0.5 second windows
  trainingData[:, (count * nSubSamples):((count + 1) * nSubSamples),:] = np.mean(np.reshape(cwtScales, (nChannels, nSubSamples, samplingRate, nf * 2)), axis=2)
  count = count + 1

  # move on to the next time step
  currentWindowStartTime = currentWindowStartTime + windowOffset
  currentWindowEndTime = currentWindowStartTime + windowLength

# save the training data
outfileName = outfilePath + 'trainingData'
np.savez(outfileName, trainingData)
outfileList.append(outfileName)

# Clustering
trainingData = scale(np.reshape(trainingData, (nChannels * nSubSamples, nf * 2)), copy=False)
nClusters = 4

# K-means
kmeans = KMeans(init='k-means++', n_clusters=nClusters, n_jobs=-1, n_init=10)
kmeans.fit(trainingData)
trainingLabels = np.reshape(kmeans.labels_, (nChannels, -1))
outfileName = outfilePath + 'kmeansClusterLabels' 
np.savez(outfileName, trainingLabels)
outfileList.append(outfileName)
outfileName = outfilePath + 'kmeansClusterCenters' 
np.savez(outfileName, kmeans.cluster_centers_)
outfileList.append(outfileName)

# dump the estimator
from sklearn.externals import joblib
outfileName = outfilePath + 'kmeans.pkl'
joblib.dump(kmeans, outfileName)
outfileList.append(outfileName)

# Hierarchical clustering 
aggloCluster = AggloCluster(n_clusters=nClusters, affinity='euclidean', compute_full_tree='false', linkage='average')
aggloCluster.fit(trainingData)
trainingLabels = np.reshape(aggloCluster.labels_, (nChannels, -1))
outfileName = outfilePath + 'aggloClusterLabels' 
np.savez(outfileName, trainingLabels)
outfileList.append(outfileName)

# dump the estimator
outfileName = outfilePath + 'aggloCluster.pkl'
joblib.dump(aggloCluster, outfileName)
outfileList.append(outfileName)

