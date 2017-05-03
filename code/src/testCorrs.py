import numpy as np
import corrs

longFct = np.array([0,1.5,2,3,0.4],dtype=np.float32)
shortFct = np.array([2,3,3,2.5,0.1],dtype=np.float32)
myCorrCoeff = corrs.correlationCoeff(longFct, shortFct, longFct.size)
print('myCorrCoeff '+str(myCorrCoeff))

correlation = np.zeros(nLags,dtype = np.int32)
#flag = corrs.oneBitXCorr(longFct,shortFct,correlation)
