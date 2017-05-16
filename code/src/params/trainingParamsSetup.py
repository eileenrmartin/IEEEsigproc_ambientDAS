import sys
sys.path.append('..')
from fileSet import *
from regularFileSet import *
import datetime as dt
import subprocess

hoursPerJob = 24 # do roughly one day of work per job
secondsPerWindowWidth = 310
secondsPerWindowOffset = 300
xCorrMaxTimeLagSeconds = 3.0
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0
outfilePath = '/scratch/'
outfileListFileStart = 'inter_results/outfileList'
srcChList = 'sourceList.txt'

# keep these file sets in order
fileSetNames = ['fs1','fs2']
fileSetStartTimes = {'fs1':dt.datetime(2016,9,3,0,0,54,932000),'fs2':dt.datetime(2016,9,7,1,6,54,932000)}
fileSetEndTimes = {'fs1':dt.datetime(2016,9,7,0,8,54,932000),'fs2':dt.datetime(2016,9,9,23,59,54,932000)}
fileSetSecPerFile = 60

fileSets = {}
# file name parts should start with /data/biondo/DAS/ if done on cees machines directly, but just /data/ if in Docker
parts = ['/data/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']
days = [] # to contain tuples of (dt.date,[fileSetNameIncludingDate,anotherFileSetNameIncludingDate])

for fsn in fileSetNames:
	startTime = fileSetStartTimes[fsn]
	endTime = fileSetEndTimes[fsn]
	secondsPerFile = fileSetSecPerFile
	nFilesIn = int((endTime-startTime).total_seconds())/secondsPerFile
	fileSets[fsn] = regularFileSet(parts,startTime.year, startTime.month, startTime.day, startTime.hour, startTime.minute, startTime.second, int(0.001*startTime.microsecond),secondsPerFile,nFilesIn)
	# record which days are in this file set (since jobs will be divied up by day)
	thisDay = dt.date(startTime.year, startTime.month, startTime.day)
	lastDay = dt.date(endTime.year, endTime.month, endTime.day)
	while thisDay <= lastDay:
		if(len(days) > 0):
			if(days[-1][0] == thisDay): # if previous file set included this day
				days[-1][1].append(fsn)
			else:
				days.append((thisDay,[fsn]))
		else:
			days.append((thisDay,[fsn]))
		thisDay = thisDay + dt.timedelta(days=1)
	
jobCounter = 1

def stringStartTime(startJobTime):
	st = startJobTime # start time a dt.datetime object
	return str(st.year)+','+str(st.month)+','+str(st.day)+','+str(st.hour)+','+str(st.minute)+','+str(st.second)+','+str(int(st.microsecond))

for day in days:
	# for each day create a job or a few jobs	
	startJobTimes = [] # holds list of first time of first file for each file set for each day
	nFilesPerJob = [] # holds list of number of files for each file set for each day
	thisDayFileSetNames = day[1]
	for fsn in thisDayFileSetNames:
		thisFileSet = fileSets[fsn]
		thisStartFile = thisFileSet.getFirstFileStartingThisDay(day[0]) # get first file that starts during this day in this file set
		thisEndFile = thisFileSet.getLastFileStartingThisDay(day[0]) # get last file that ends during this day in this file set
		# get times and number of files from the file names 
		thisStartJobTime = thisFileSet.getTimeFromFilename(thisStartFile)
		thisEndJobTime = thisFileSet.getTimeFromFilename(thisEndFile) # actually the start time of the last file
		if(thisStartJobTime < thisEndJobTime):
			thisJobNFiles = 1+int((thisEndJobTime-thisStartJobTime).total_seconds()/fileSetSecPerFile)
			# record start time and number of files for this file set on this day
			startJobTimes.append(thisStartJobTime)
			nFilesPerJob.append(thisJobNFiles)

	if(len(nFilesPerJob) > 0):
		# make the folder and open up the file to write to
		returnflag = subprocess.call(["mkdir","trainingparams"+str(jobCounter)])
		paramsFileName = "params"+str(jobCounter)+"/params.py"
		paramsFile = open(paramsFileName,'w')
		# start writing the file
		paramsFile.write('import datetime as dt \n')
		# write list of starting times for file sets within the day
		startTimeString = 'startTimes = ['
		for i,sjt in enumerate(startJobTimes):
			if i > 0:
				startTimeString = startTimeString + ', '
			startTimeString = startTimeString + 'dt.datetime('+stringStartTime(sjt)+')'
		paramsFile.write(startTimeString+'] \n')
		# other info about how to partition time
		paramsFile.write('secondsPerFile = '+str(fileSetSecPerFile)+'\n')
		paramsFile.write('secondsPerWindowWidth = '+str(secondsPerWindowWidth)+'\n')
		paramsFile.write('secondsPerWindowOffset = '+str(secondsPerWindowOffset)+'\n')
		paramsFile.write('xCorrMaxTimeLagSeconds = '+str(xCorrMaxTimeLagSeconds)+'\n')
		# write list of number of files for file sets within the day
		nFilesString = 'nFiless = ['
		for i,nfj in enumerate(nFilesPerJob):
			if i > 0:
				nFilesString = nFilesString + ', '
			nFilesString = nFilesString + str(nfj)
		paramsFile.write(nFilesString+'] \n')
		# other info about where to get and write intermediate results, which channels to use and which frequency bands to use
		paramsFile.write("outfilePath = '"+outfilePath+"'\n")
		paramsFile.write("outfileListFile = '"+outfileListFileStart+str(jobCounter)+".txt'\n")
		paramsFile.write("srcChList = '"+srcChList+"'\n")
		paramsFile.write("startCh = "+str(startCh)+"\n")
		paramsFile.write("endCh = "+str(endCh)+"\n")
		paramsFile.write("minFrq = "+str(minFrq)+"\n")
		paramsFile.write("maxFrq = "+str(maxFrq)+"\n")
		paramsFile.close()
	

		# increase the counter of jobs being created
		jobCounter = jobCounter + 1
