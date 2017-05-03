#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cfloat>

double _correlationCoeff(float *longTermFct, float *shortTermFct, int n){
  /// Calculates the zero time lag correlation 
  /// but does so with blocked sums (so no overflow)
  /// n is length of those arrays

  double corrCoeff = 0.0;
  float partialSum = 0.0;

  // break arrays into blocks
  for(int i=0; i<n; ++i){
    partialSum += longTermFct[i] * shortTermFct[i];
    if(std::abs(partialSum) > 0.25*FLT_MAX){  // if the partial sum is getting too big for floats, throw this sum into the double counter
      corrCoeff += double(partialSum);
      partialSum = 0;
    }
  }
  // add any stragler terms at the end
  corrCoeff += double(partialSum);

  return corrCoeff;
}

static PyObject* correlationCoeff(PyObject* self, PyObject* args){
    PyObject  *longTermArg=NULL, *shortTermArg=NULL;
    float  *longTermFct=NULL, *shortTermFct=NULL;
    PyObject *output=NULL;
    int nSamples;

    if (!PyArg_ParseTuple(args,"OOi", &longTermArg, &shortTermArg, &nSamples)) return NULL;
    longTermFct = (float *)PyArray_GETPTR1(longTermArg,0);
    if (longTermFct == NULL) return NULL;
    shortTermFct = (float *)PyArray_GETPTR1(shortTermArg,0);
    if (shortTermFct == NULL) return NULL;

    //int nd = PyArray_NDIM(longTermFct); // number fo dimensions

    double c = _correlationCoeff(longTermFct,shortTermFct,nSamples);

    Py_DECREF(longTermFct);
    Py_DECREF(shortTermFct);

	return Py_BuildValue("d", c);
}

static PyMethodDef CorrelationCoeffMethods[] = {
	{"correlationCoeff",correlationCoeff,METH_VARARGS, "Calculate the 0 lag correlation of two time series"},
	{NULL,NULL,0,NULL}

};

PyMODINIT_FUNC
initcorrs(void){
	(void) Py_InitModule("corrs",CorrelationCoeffMethods);
	import_array();
}

