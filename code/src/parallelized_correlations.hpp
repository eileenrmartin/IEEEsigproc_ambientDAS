#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include <vector>
#include <iostream>

void onePairOneBitXCorr(float *virtualSrcVec, float *aReceiver, int *xcorrOfPair, int nSamples, int nLags){
	// create thresholded data (should do this outside if you're doing many virtual sources)
	signed char vs [nSamples];
	signed char r [nSamples];
	for(int i=0; i<nSamples; ++i){
		if(virtualSrcVec[i] >= 0){
			vs[i] = (signed char)1;
		} else{
			vs[i] = (signed char)-1;
		}
		if(aReceiver[i] >= 0){
			r[i] = (signed char)1;
		} else{
			r[i] = (signed char)-1;
		}
	}		

	// for each lag, multiply and add to cross correlation for the corresponding tau value, then slide along
	int sumWidth = nSamples - 2*nLags;
	for(int i=-nLags; i<=nLags; ++i){
		int startSample = i + nLags;
		xcorrOfPair[startSample] = 0; 
		for(int j=0; j<sumWidth; ++j){
			xcorrOfPair[startSample] += int(vs[startSample+j]*r[nLags+j]);		
		}
	}
}


int par_oneBitXcorr(float *virtualSrcVec, int nSamples, float *receiverMat, int nRecs, int *xcorrMat, int nLags){
	// xCorrMat should be preallocated as nRecs x nLags 
	// slow dimension of receiverMat shoudl be nRecs long and fast dimension nSamples
	// virtualSrcVec shoudl be 1+2*nSamples long

	// do the correlations of the one bit thresholded data
	tbb::parallel_for( size_t(0), size_t(nRecs), size_t(1), [=](size_t i){onePairOneBitXCorr(&virtualSrcVec[0], &receiverMat[i*nSamples], &xcorrMat[i*(nLags*2+1)], nSamples, nLags);});
	return 1;
}
