#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <float.h>
#include "parallelized_correlations.hpp"

static PyObject* crosscorrs_func(PyObject* self, PyObject* args){
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

    par_oneBitXcorr(virtualSrcVec, nSamples, receiverMat, nRecs, xcorrMat, nLags);
    
    Py_DECREF(virtualSrcVec);
    Py_DECREF(receiverMat);
    Py_DECREF(xcorrMat);
 
    int flag = 1;
    return Py_BuildValue("i",flag);
}

static PyMethodDef CrosscorrsMethods[] = {
	{"crosscorrs_func", crosscorrs_func, METH_VARARGS, "Calculate the crosscorrelation of two time series"},
	{NULL,NULL,0,NULL}

};

PyMODINIT_FUNC
initcrosscorrs_module(void){
	(void) Py_InitModule("crosscorrs_module",CrosscorrsMethods);
	import_array();
}

