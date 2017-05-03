import numpy as np
import corrs_module

longFct = np.array([0,1.5,2,3,0.4],dtype=np.float32)
shortFct = np.array([2,3,3,2.5,0.1],dtype=np.float32)


print('at the beginning before calling C function')
print('longFct '+str(longFct))
print('shortFct '+str(shortFct))


myCorrCoeff = corrs_module.corrs_func(longFct, shortFct, longFct.size)

print('my 0 lag correlation coefficient is '+str(myCorrCoeff))
print('Right after calling the C function:')
print('longFct '+str(longFct))
print('shortFct '+str(shortFct))

print('Now I convert all nans to 0')
longFct = np.nan_to_num(longFct)
shortFct = np.nan_to_num(shortFct)
print('longFct '+str(longFct))
print('shortFct '+str(shortFct))


#nLags = 2
#correlation = np.zeros(nLags,dtype = np.int32)
#flag = corrs.oneBitXCorr(longFct,shortFct,correlation)
