#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <float.h>
//#include "parallelized_correlations.hpp"
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include <vector>

void onePairOneBitXCorr(float *virtualSrcVec, float *aReceiver, int *xcorrOfPair, int nSamples, int nLags){
	// create thresholded data (should do this outside if you're doing many virtual sources)
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
	int sumWidth = nSamples - 2*nLags;
	for(int i=-1*nLags; i<=nLags; ++i){
		int startSample = i + nLags;
		int endSamples = startSample + sumWidth;
		xcorrOfPair[i] = 0; 
		for(int j=0; j<sumWidth; ++j){
			xcorrOfPair[i] += int(vs[startSample+j]*r[nLags+j]);			
		}
	}
}


int par_oneBitXcorr(float *virtualSrcVec, int nSamples, float *receiverMat, int nRecs, int *xcorrMat, int nLags){
	// xCorrMat should be preallocated as nRecs x nLags 
	// slow dimension of receiverMat shoudl be nRecs long and fast dimension nSamples
	// virtualSrcVec shoudl be 1+2*nSamples long

	// do the correlations of the one bit thresholded data
	tbb::parallel_for( size_t(1), size_t(nRecs+1), size_t(1), [=](size_t i){onePairOneBitXCorr(&virtualSrcVec[0], &receiverMat[i*nSamples], &xcorrMat[i*nLags], nSamples, nLags);});
	return 1;
}


static PyObject* crosscorr_func(PyObject* self, PyObject* args){
    PyObject  *virtualSrcArg=NULL, *receiverArg=NULL, *xcorrArg=NULL;
    float  *virtualSrcVec=NULL, *receiverMat=NULL;
    int *xcorrMat=NULL;
    int nSamples, nRecs, nLags;

   
    if (!PyArg_ParseTuple(args,"OiOiOi", &virtualSrcArg, &nSamples, &receiverArg, &nRecs, &xcorrArg, &nLags)) return NULL;
    
    virtualSrcVec = (float *)PyArray_GETPTR1(virtualSrcArg,0);
    if (virtualSrcVec == NULL) return NULL;
    receiverMat = (float *)PyArray_GETPTR1(receiverArg,0);
    if (receiverMat == NULL) return NULL;
    xcorrMat = (int *)PyArray_GETPTR1(xcorrArg,0);
    if (xcorrMat == NULL) return NULL;

    int flag = par_oneBitXcorr(virtualSrcVec, nSamples, receiverMat, nRecs, xcorrMat, nLags);
    
    Py_DECREF(virtualSrcVec);
    Py_DECREF(receiverMat);
    Py_DECREF(xcorrMat);
 
    return Py_BuildValue("i",flag);
}

static PyMethodDef CrosscorrMethods[] = {
	{"crosscorr_func", crosscorr_func, METH_VARARGS, "Calculate the crosscorrelation of two time series"},
	{NULL,NULL,0,NULL}

};

PyMODINIT_FUNC
initcrosscorr_module(void){
	(void) Py_InitModule("crosscorr_module",CrosscorrMethods);
	import_array();
}

