#include <Python.h>
#include "/usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h"
#include <float.h>

double _correlationCoeff(float *longTermFct, float *shortTermFct, int n){
  /// Calculates the zero time lag correlation 
  /// but does so with blocked sums (so no overflow)
  /// n is length of those arrays

  double corrCoeff = 0.0;
  float partialSum = 0.0;

  // break arrays into blocks
  for(int i=0; i< n; ++i){
    partialSum += longTermFct[i] * shortTermFct[i];
    if(i % 10000 == 1){  // if the partial sum might get too big for floats, throw this sum into the double counter
      corrCoeff += double(partialSum);
      partialSum = 0;
    }
  }
  // add any stragler terms at the end
  corrCoeff += double(partialSum);

  return corrCoeff;
}

static PyObject* corrs_func(PyObject* self, PyObject* args){
    PyObject  *longTermArg=NULL, *shortTermArg=NULL;
    float  *longTermFct=NULL, *shortTermFct=NULL;
    int nSamplesArg;
    double c;
   
    if (!PyArg_ParseTuple(args,"OOi", &longTermArg, &shortTermArg, &nSamplesArg)) return NULL;
    
    longTermFct = (float *)PyArray_GETPTR1(longTermArg,0);
    if (longTermFct == NULL) return NULL;
    shortTermFct = (float *)PyArray_GETPTR1(shortTermArg,0);
    if (shortTermFct == NULL) return NULL;

    c = _correlationCoeff(longTermFct,shortTermFct,nSamplesArg);
    
    Py_DECREF(longTermFct);
    Py_DECREF(shortTermFct);
 
    return Py_BuildValue("d", c);
}

static PyMethodDef CorrsMethods[] = {
	{"corrs_func", corrs_func, METH_VARARGS, "Calculate the 0 lag correlation of two time series"},
	{NULL,NULL,0,NULL}

};

PyMODINIT_FUNC
initcorrs_module(void){
	(void) Py_InitModule("corrs_module",CorrsMethods);
	import_array();
}

