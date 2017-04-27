import datetime as dt

startTime = dt.datetime(2016,9,4,5,0,54,932*1000)
secondsPerFile = 60
secondsPerWindowWidth = 300
secondsPerWindowOffset = 150
xCorrMaxTimeLagSeconds = 3.0
nFiles = 240
outfilePath = '/scratch/ermartin/oneBitXCorr/'
outfileListFile = '/home/ermartin/StanfordDASArray/inter_results/outfileList1.txt'
srcChList = 'sourceList.txt'
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0