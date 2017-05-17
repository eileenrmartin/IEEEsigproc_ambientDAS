import numpy as np
import datetime as dt
import struct
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AggloCluster

paramsPath = sys.argv[1]
sys.path.append(paramsPath)
from params import *

srcFile = open(srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
nChannels = endCh - startCh + 1
outfileList = []

ndays = 7
outfileListFile = outfileListFile[:-5]
files = []
nf = 30
sps = 50
samplingRate = 25
nSamples = secondsPerWindowOffset * int(sps / samplingRate)

for day in range(1, ndays + 1):
  fileListName = outfileListFile + str(day) + '.txt'
  f = open(fileListName,'r')
  fileList = f.readlines()
  dayFiles = [(file.strip('\n')) for file in fileList]
  files.extend(dayFiles)
  f.close()

nfiles = len(files)

trainingData = np.empty((nChannels, nSamples * nfiles, nf * 2), dtype=np.float64)

for index, file in enumerate(files):
  file = outfilePath + file
  trainingData[:,(index * nSamples):((index + 1) * nSamples),:] = np.load(file)

# Clustering
trainingData = np.reshape(trainingData, (nChannels * nSamples * nfiles, -1))
scaler = StandardScaler(copy=False).fit(trainingData)
trainingData = scaler.transform(trainingData)
nClusters = 4

# dump the scaler
from sklearn.externals import joblib
outfileName = outfilePath + 'scaler.pkl'
joblib.dump(scaler, outfileName)
outfileList.append(outfileName)

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

# write the list of output file names
outfileListFile = outfileListFile + '_clustering'
outFile = open(outfileListFile,'w')
for filename in outfileList:
  outFile.write(filename + '\n')
outFile.close()
