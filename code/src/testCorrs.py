import numpy as np
import corrs_module
import onebitcrosscorr_module

longFct = np.array([0,-1.5,2,3,0.4],dtype=np.float32)
shortFct = np.array([2,3,3,2.5,0.1],dtype=np.float32)


print('at the beginning before calling C function')
print('longFct '+str(longFct))
print('shortFct '+str(shortFct))


myCorrCoeff = corrs_module.corrs_func(longFct, shortFct, longFct.size)

print('my 0 lag correlation coefficient is '+str(myCorrCoeff))
print('Right after calling the C function:')
print('longFct '+str(longFct))
print('shortFct '+str(shortFct))



#nLags = 2
#correlation = np.zeros(nLags,dtype = np.int32)
#flag = corrs.oneBitXCorr(longFct,shortFct,correlation)
longFct2 = np.array([0,-1.5,2,3,0.4],dtype=np.float32)
shortFct2 = np.array([2,3,3,2.5,0.1],dtype=np.float32)
nSamples = longFct2.size
nLags = 1
nRecs = 1
xcorr = np.zeros(2*nLags+1,dtype=np.int32)

flag = onebitcrosscorr_module.onebitcrosscorr_func(longFct2, nSamples, shortFct2, nRecs, xcorr, nLags)
print('after calling onebitcrosscorr flag '+str(flag))
print('xcorr '+str(xcorr))
print('longFct '+str(longFct2))
print('shortFct '+str(shortFct2))
