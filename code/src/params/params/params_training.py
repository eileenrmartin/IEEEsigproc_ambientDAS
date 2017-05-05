import datetime as dt 
startTime = dt.datetime(2016,9,3,0,0,54,932000) 
secondsPerFile = 60
secondsPerWindowWidth = 60 * 60 * 4 + 120
secondsPerWindowOffset = 60 * 60 * 4
xCorrMaxTimeLagSeconds = 3.0
nFiles = 1440 * 7 
outfilePath = '/scratch/'
outfileListFile = 'inter_results/outfileTraining.txt'
srcChList = 'sourceList.txt'
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0
