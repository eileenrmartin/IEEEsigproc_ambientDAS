#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <float.h>
#include "parallelized_correlations.hpp"
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"
#include <vector>
#include <iostream>

static PyObject* onebitcrosscorr_func(PyObject* self, PyObject* args){
    PyObject  *virtualSrcArg=NULL, *receiverArg=NULL, *xcorrArg=NULL;
    float  *virtualSrcVec=NULL, *receiverMat=NULL;
    int *xcorrMat=NULL;
    int nSamples, nRecs, nLags;

   
    if (!PyArg_ParseTuple(args,"OiOiOi", &virtualSrcArg, &nSamples, &receiverArg, &nRecs, &xcorrArg, &nLags)) return NULL;
    
    virtualSrcVec = (float *)PyArray_GETPTR1(virtualSrcArg,0);
    if (virtualSrcVec == NULL) return NULL;
    receiverMat = (float *)PyArray_GETPTR2(receiverArg,0,0);
    if (receiverMat == NULL) return NULL;
    xcorrMat = (int *)PyArray_GETPTR2(xcorrArg,0,0);
    if (xcorrMat == NULL) return NULL;
    std::vector<float *> recPtrs;
    std::vector<int *> xcorrPtrs;
    recPtrs.reserve(nRecs);
    xcorrPtrs.reserve(nRecs);
    for(int r=0; r<nRecs; ++r){
        recPtrs.push_back((float *)PyArray_GETPTR2(receiverArg,r,0));
        xcorrPtrs.push_back((int *)PyArray_GETPTR2(xcorrArg,r,0));	
    }
 
    int flag = par_oneBitXcorr(virtualSrcVec, nSamples, recPtrs, nRecs, xcorrPtrs,nLags);
    
    //Py_DECREF(virtualSrcArg);
    //Py_DECREF(receiverArg);
    //Py_DECREF(xcorrArg);
 
    return Py_BuildValue("i",flag);
}

static PyMethodDef OnebitcrosscorrMethods[] = {
	{"onebitcrosscorr_func", onebitcrosscorr_func, METH_VARARGS, "Calculate the crosscorrelation of two time series after one bit threshold"},
	{NULL,NULL,0,NULL}

};

PyMODINIT_FUNC
initonebitcrosscorr_module(void){
	(void) Py_InitModule("onebitcrosscorr_module",OnebitcrosscorrMethods);
	import_array();
}

