import sys
sys.path.append('..')
from fileSet import *
from regularFileSet import *
import datetime as dt
import subprocess

hoursPerJob = 24 # do roughly one day of work per job
secondsPerWindowWidth = 300
secondsPerWindowOffset = 150
xCorrMaxTimeLagSeconds = 3.0
startCh = 15
endCh = 300
minFrq = 0.2
maxFrq = 24.0
outfilePath = '/scratch/'
outfileListFileStart = 'inter_results/outfileList'
srcChList = 'sourceList.txt'

# keep these file sets in order
fileSetNames = ['fs1','fs2','fs3','fs4']
fileSetStartTimes = {'fs1':dt.datetime(2016,9,3,0,0,54,932),'fs2':dt.datetime(2016,9,12,13,27,43,376),'fs3':dt.datetime(2016,9,17,16,50,43,496),'fs4':dt.datetime(2016,9,18,18,42,0,848)}
fileSetEndTimes = {'fs1':dt.datetime(2016,9,12,13,5,54,932),'fs2':dt.datetime(2016,9,17,16,38,43,376),'fs3':dt.datetime(2016,9,18,18,3,43,496),'fs4':dt.datetime(2016,10,5,0,0,0,848)}
fileSetSecPerFile = {'fs1':60,'fs2':60,'fs3':60,'fs4':60}

fileSets = []
parts = ['/data/biondo/DAS/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']
days = [] # to contain tuples of (dt.date,[fileSetNameIncludingDate,anotherFileSetNameIncludingDate])

for fsn in fileSetNames:
	startTime = fileSetStartTimes[fsn]
	endTime = fileSetEndTimes[fsn]
	secondsPerFile = fileSetSecPerFile[fsn]
	nFilesIn = int((endTime-startTime).total_seconds())/secondsPerFile
	fileSets.append(rfs.regularFileSet(parts,startTime.year, startTime.month, startTime.day, startTime.hour, startTime.minute, startTime.second, int(0.001*startTime.microsecond),secondsPerFile,nFilesIn))
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

print(fileSets)
	
jobCounter = 2

def stringStartTime(startJobTime):
	st = startJobTime # start time a dt.datetime object
	return str(st.year)+','+str(st.month)+','+str(st.day)+','+str(st.hour)+','+str(st.minute)+','+str(st.second)+','+str(int(0.001*st.microsecond))

for day in days:
	# for each day create a job or a few jobs	
	startJobTimes = [] # ****
	nFilesPerJob = [] # ****
	thisDayFileSetNames = day[1]
	for fset in thisDayFileSetNames:
		# ******record number of files and starting time for this file set on this day

	# make the folder and open up the file to write to
	returnflag = subprocess.call(["mkdir","params"+str(jobCounter)])
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
	paramsFile.write('secondsPerFile = '+str(nFilesPerJob)+'\n')
	paramsFile.write('secondsPerWindowWidth = '+str(secondsPerWindowWidth)+'\n')
	paramsFile.write('secondsPerWindowOffset = '+str(secondsPerWindowOffset)+'\n')
	paramsFile.write('xCorrMaxTimeLagSeconds = '+str(xCorrMaxTimeLagSeconds)+'\n')
	# write list of number of files for file sets within the day
	nFilesString = 'nFiless = ['
	for i,nfj in enumerate(job[1]):
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
