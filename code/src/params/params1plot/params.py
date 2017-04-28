import datetime as dt

startTime = dt.datetime(2016,9,12,12,50,54,932*1000)
secondsPerFile = 60
secondsPerWindowWidth = 300
secondsPerWindowOffset = 150
xCorrMaxTimeLagSeconds = 3.0
nFiles = 9
outfilePath = '/scratch/'
outfileListFiles = ['inter_results/outfileList1.txt']
srcChList = 'sourceList.txt'
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0
recLocationsFilename = 'inter_results/passive_channel_UTM_coords.txt'
