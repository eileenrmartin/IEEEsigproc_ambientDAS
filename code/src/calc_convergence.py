import numpy as np
import sys
import numpy.linalg as la

windowHrs = [1,3,6,12,24,48,96,192,384] # windows over which to look at convergence
paramsPath = sys.argv[1]
filteredFlag = sys.argv[2]
startParams = int(sys.argv[3])
lastParams = int(sys.argv[4])
sys.path.append(paramsPath+str(startParams))
import params
# read the source channels
srcFile  = open(params.srcChList,'r')
srcChannelsStrings = srcFile.readlines()
srcFile.close()
srcChannels = [int(ch) for ch in srcChannelsStrings]
# read which channels are receivers
startCh = params.startCh
endCh = params.endCh
nChannels = endCh-startCh+1
# read cross-correlation lag extents
nLags = 0 # set later from size of matrices
outfilePath = params.outfilePath

sys.path.remove(paramsPath+str(startParams))


parts=['/data/','year4','/','month','/','day','/cbt_processed_','year4','month','day','_','hour24start0','minute','second','.','millisecond','+0000.sgy']

listOfAllXCorrs = []
for p in range(startParams,lastParams+1):
    # get all info for this subset of cross correlations
    sys.path.append(paramsPath+str(p))
    reload(params)
   
    # get the list of all the output files
    outFileList = open(params.outfileListFile+'_'+filteredFlag+'.txt','r')
    listOfAllFilesBasic = outFileList.readlines()
    listOfAllDataFiles = [(f.strip('\n'))+'_data.npz' for f in listOfAllFilesBasic]
    listOfAllHeaderFiles = [(f.strip('\n'))+'_headers.txt' for f in listOfAllFilesBasic]
    outFileList.close()

    # read xcorrs for the day
    filename = listOfAllDataFiles[0] # theres just one in there
    todayXCorrs = np.load(filename)['arr_0']
    nHrs = xCorr.shape[0]
    if(nHrs > 24):
        todayXCorrs = todayXCorrs[:24,:,:,:]
    nLags = todayXCorrs.shape[3]

    # append today's xcorrs matrix to the list of all xcorrs
    listOfAllXCorrs.append(todayXCorrs)

    sys.path.remove(paramsPath+str(p))

# assemble the xcorrs matrices for all days into one array
nTotalHrs = 24*(lastParams-startParams+1)
if (lastParams == 30): # there's a mistake that chops this day off early
    nTotalhrs = nTotalHrs-9
allXCorrs = np.zeros((nTotalHrs,len(srcChannels),nChannels,nLags))
hrIdx = 0
for xcorr in listOfAllXCorrs:
    nHrs = xcorr.shape[0]
    allXCorrs[hrIdx:hrIdx+nHrs,:,:,:] = xcorr
    hrIdx = hrIdx+nHrs


# get the average long term cross correlation during this time
longTermAvg = np.sum(allXCorrs,axis=0)/nTotalHrs
longTermAvgXCorrFilename = outfilePath+'longTermAvgXCorr_'+str(startParams)+'_to_'+str(lastParams)
np.savez(longTermAvgXCorrFilename,longTermAvg)


def zeroLagCorr(longterm,shortterm):
    corr = np.dot(longterm,shortterm)/(la.norm(longterm)*la.norm(shortterm))
    return corr

# for each of the window widths calculate and save the convergence correlation metric
for window in windowHrs:
    print('calculating convergence of '+str(window)+' hour subsets')

    # calculate a set of subset of hours averages, continuous hours the number of hours specified
    nWindows = nTotalHrs-window+1
    subAvgXCorr = np.zeros((nWindows,len(srcChannels),nChannels,nLags),dtype=np.float32)
    for w in range(window):
        subAvgXCorr = subAvgXCorr + allXCorrs[w::window,:,:,:]
    
    # create matrix of zero time lag correlations to measure convergence to long term avg
    RC = np.zeros((nWindows,len(srcChannels),nChannels),dtype=np.float32)
    for w in range(nWindows):
        for s in range(len(srcChannels)):
            longtermSrc = longTermAvg[s,:,:]
            shorttermSrc = subAvgXCorr[w,s,:,:]
            for r in range(nChannels):
                RC[w,s,r] = zeroLagCorr(longtermSrc[r,:],shortTermSrc[r,:])

    thisWindowOutfile = outfilePath+'convergence_of_'+str(window)+'_hr_windows'
    np.savez(thisWindowOutfile,RC)
   
