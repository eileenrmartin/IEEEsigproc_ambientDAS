import numpy as np
import matplotlib.pyplot as plt
import sys

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
minLag = -params.xCorrMaxTimeLagSeconds
maxLag = -minLag
outfilePath = '/scratch/' #params.outfilePath

sys.path.remove(paramsPath+str(startParams))

ndays = -startParams+lastParams+1

# plot overall average cross correlations
filename = outfilePath+'longTermAvgXCorr_'+str(startParams)+'_to_'+str(lastParams)+'_'+filteredFlag+'.npz'
avgXCorr = np.load(filename)['arr_0']
nLags = avgXCorr.shape[2]
avgXCorr = avgXCorr[:,:,1+nLags/2:] + np.flip(avgXCorr[:,:,:nLags/2],axis=2)
print(avgXCorr.shape)
for ich,ch in enumerate(srcChannels):
    # normalize channel-wise
    for i in range(startCh,endCh+1):
        norm = np.sum(np.absolute(avgXCorr[ich,i-startCh,:]))
        #if(norm > 0):
        #    avgXCorr[ich,i-startCh,:] = avgXCorr[ich,i-startCh,:]/norm
    # clip and plot
    clipVal = np.percentile(np.absolute(avgXCorr[ich,:,:]),99)
    plt.imshow(avgXCorr[ich,:,:],aspect='auto',interpolation='nearest',vmin=-clipVal,vmax=clipVal,cmap=plt.get_cmap('seismic'),extent=[0,maxLag,endCh,startCh])
    plt.ylabel('channel (8 m/channel)')
    plt.xlabel('time lag (seconds)')
    plt.title(str(ndays)+' day '+filteredFlag+' correlation for virtual source ch. '+str(ch))
    filename = outfilePath+'long_term_avg_oneBit_xcorr_'+filteredFlag+'_srcCh_'+str(ch)+'.pdf'
    plt.savefig(filename)
    plt.clf()


# plot correlation trends
RCs = []
for ich,ch in enumerate(srcChannels):

    for w in windowHrs:
        thisWindowCorrFile = outfilePath+'convergence_of_'+str(w)+'_hr_windows_'+filteredFlag+'.npz'
        RC = np.load(thisWindowCorrFile)['arr_0']
        nWindows = RC.shape[0]
        RCs.append(RC)
  
        # show whether for a particular virutal source and window width, if some parts of the array are more converged than others
        plt.imshow(RC[:,ich,:],aspect='auto',interpolation='nearest',cmap=plt.get_cmap('inferno'),vmin=0,vmax=1,extent=[startCh,endCh,ndays,1])
        plt.ylabel('day that window starts')
        plt.xlabel('channel (8 m/channel)')
        plt.title(str(w)+' hours, RC for correlations against channel '+str(ch))
        plt.colorbar()
        filename = outfilePath+'RC_'+filteredFlag+'_srcCh_'+str(ch)+'_'+str(w)+'_hrs_windows.pdf'
        plt.savefig(filename)
        plt.clf()
    
    
