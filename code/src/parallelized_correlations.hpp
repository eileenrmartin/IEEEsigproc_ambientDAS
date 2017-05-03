#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include <vector>

void onePairOneBitXCorr(float *virtualSrcVec, float *aReceiver, int *xcorrOfPair, int nSamples, int nLags){
	// create thresholded data (shoudl do this outside if you're doing many virtual sources)
	std::vector<char> vs, r;
	vs.resize(nSamples);
	r.resize(nSamples);
	for(int i=0; i<nSamples; ++i){
		if(virtualSrcVec[i] >= 0){
			vs[i] = 1;
		} else{
			vs[i] = -1;
		}
		if(aReceiver[i] >= 0){
			r[i] = 1;
		} else{
			r[i] = -1;
		}
	}		

	// for each lag, multiply and add to cross correlation for the corresponding tau value, then slide along
	int sumWidth = nSamples - 2*nLagSamples;
	for(int i=-1*nLagSamples; i<=nLagSamples; ++i){
		int startSample = i + nLagSamples;
		int endSamples = startSample + sumWidth;
		xcorrOfPair[i] = 0; 
		for(int j=0; j<sumWidth; ++j){
			xcorrOfPair[i] += int(vs[startSample+j]*r[nLagSamples+j]);			
		}
	}
}

int par_oneBitXcorr(float *virtualSrcVec, int nSamples, float *receiverMat, int nRecs, int *xcorrMat, int nSamples, int nLags){
	// xCorrMat should be preallocated as nRecs x nLags 
	// slow dimension of receiverMat shoudl be nRecs long and fast dimension nSamples
	// virtualSrcVec shoudl be 1+2*nSamples long
	tbb::parallel_for( size_t(1), nRecs+1, size_t(1), [=](size_t i){onePairOneBitXCorr(virtualSrcVec[0], receiverMat[i*nSamples], xcorrMat[i*nLags], nSamples, nLags);});
}
