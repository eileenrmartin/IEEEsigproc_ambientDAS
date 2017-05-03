import numpy as np
import struct

nTxtFileHeader = 3200
nBinFileHeader = 400
nTraceHeader = 240
def readTrace(infile,nSamples,dataLen,traceNumber,endian,startSample,nSamplesToRead):
    '''infile is .sgy, nSamples is the number of samples per sensor, and traceNumber is the sensor number (start with 1),dataLen is number of bytes per data sample'''

    fin = open(infile, 'rb') # open file for reading binary mode
    startData = nTxtFileHeader+nBinFileHeader+nTraceHeader+(traceNumber-1)*(nTraceHeader+dataLen*nSamples)+startSample*dataLen
    fin.seek(startData)
    thisTrace = np.zeros(nSamplesToRead)
    thisDataBinary = fin.read(nSamplesToRead*dataLen)
    thisDataArray = struct.unpack_from(endian+'f',thisDataBinary)
    for i in range(nSamplesToRead):
       	# was >f before
       	thisTrace[i] = struct.unpack(endian+'f',fin.read(dataLen))[0]
    fin.close()
    return thisTrace

