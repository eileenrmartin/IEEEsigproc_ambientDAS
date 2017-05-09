import numpy as np
import struct
import matplotlib.pyplot as plt

nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240
def readTrace(infile,nSamples,dataLen,traceNumber,endian,startSample,nSamplesToRead):
    '''infile is .sgy, nSamples is the number of samples per sensor, and traceNumber is the sensor number (start with 1),dataLen is number of bytes per data sample'''

    fin = open(infile, 'rb') # open file for reading binary mode
    startData = nTxtFileHeader+nBinFileHeader+nTraceHeader+(traceNumber-1)*(nTraceHeader+dataLen*nSamples)+startSample*dataLen
    fin.seek(startData)
    thisDataBinary = fin.read(nSamplesToRead*dataLen) # read binary bytes from file
    fin.close()
    thisDataArray = struct.unpack_from(endian+('f')*nSamplesToRead,thisDataBinary) # get data as a tuple of floats
    return np.asarray(thisDataArray,dtype=np.float32)

