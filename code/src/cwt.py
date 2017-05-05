import numpy as np

__all__ = ["cwt", "icwt", "autoscales", "fourier_from_scales",
           "scales_from_fourier"]

PI2 = 2 * np.pi


def normalization(s, dt):
  return np.sqrt((PI2 * s) / dt)


def morletft(s, w, w0, dt):
  """Fourier tranformed morlet function
  
  Parameters:
    s : scales
    w : angular frequencies
    w0 : omega0 (frequency)
    dt : time step

  Returns:
    (normalized) fourier transformed morlet function
  """
  
  p = 0.75112554446494251 # pi**(-1.0/4.0)
  wavelet = np.zeros((s.shape[0], w.shape[0]))
  pos = w > 0

  for i in range(s.shape[0]):
      n = normalization(s[i], dt)
      wavelet[i][pos] = n * p * np.exp(-(s[i] * w[pos] - w0)**2 / 2.0)
      
  return wavelet


def angularfreq(N, dt):
  """Compute angular frequencies

  Parameters:   
    N : integer
      number of data samples
    dt : float
      time step
  
  Returns:
    angular frequencies : 1d numpy array
  """
  
  N2 = N / 2.0
  w = np.empty(N)

  for i in range(w.shape[0]):
    if i <= N2:
      w[i] = (PI2 * i) / (N * dt)
    else:
      w[i] = (PI2 * (i - N)) / (N * dt)

  return w


def autoscales(N, dt, dj, wf, w0):
  """Compute scales as fractional power of two

  Parameters:
    N : integer
      number of data samples
    dt : float
      time step
    dj : float
      scale resolution
    wf : string
      wavelet function ('morlet')
    w0 : float
      omega0

  Returns:
    scales : 1d numpy array
  """
     
  if wf == 'morlet':
    s0 = (dt * (w0 + np.sqrt(2 + w0**2))) / (PI2)
  else:
    raise ValueError('wavelet function not available')

  J = np.floor(dj**-1 * np.log2((N * dt) / s0))
  s = np.empty(int(J + 1))

  for i in range(s.shape[0]):
    s[i] = s0 * 2**(i * dj)

  return s



def fourier_from_scales(scales, wf, w0):
  """Compute the equivalent fourier period
  from scales.
  
  Parameters:
    scales : 1d numpy array
      scales
    wf : string ('morlet')
      wavelet function
    w0 : float
      wavelet function parameter ('omega0')
  
  Returns:
     fourier wavelengths
  """

  scales_arr = np.asarray(scales)

  if wf == 'morlet':
    return  (4 * np.pi * scales_arr) / (w0 + np.sqrt(2 + w0**2))
  else:
    raise ValueError('wavelet function not available')


def scales_from_fourier(f, wf, w0):
  """Compute scales from fourier period

  Parameters:
    f : 1d numpy array
      fourier wavelengths
    wf : string ('morlet')
      wavelet function
    w0 : float
      wavelet function parameter ('omega0')
  
  Returns:
    scales
  """

  f_arr = np.asarray(f)

  if wf == 'morlet':
    return (f_arr * (w0 + np.sqrt(2 + w0**2))) / (4 * np.pi)
  else:
    raise ValueError('wavelet function not available')


def cwt(x, dt, scales, wf='morlet', w0=2):
  """Continuous Wavelet Tranform

  Parameters:
    x : 1d array_like object
      data
    dt : float
      time step
    scales : 1d array_like object
      scales
    wf : string ('morlet')
      wavelet function
    w0 : float
      wavelet function parameter ('omega0')
          
  Returns:
    X : 2d numpy array
      transformed data
  """

  x_arr = np.asarray(x) - np.mean(x)
  scales_arr = np.asarray(scales)

  if x_arr.ndim is not 1:
    raise ValueError('x must be an 1d numpy array')

  if scales_arr.ndim is not 1:
    raise ValueError('scales must be an 1d numpy array')

  w = angularfreq(N=x_arr.shape[0], dt=dt)
      
  if wf == 'morlet':
    wft = morletft(s=scales_arr, w=w, w0=w0, dt=dt)
  else:
    raise ValueError('wavelet function is not available')
  
  X_ARR = np.empty((wft.shape[0], wft.shape[1]), dtype='complex128')
      
  x_arr_ft = np.fft.fft(x_arr)

  for i in range(X_ARR.shape[0]):
      X_ARR[i] = np.fft.ifft(x_arr_ft * wft[i])
  
  return X_ARR


def icwt(X, dt, scales, wf='morlet', w0=2):
  """Inverse Continuous Wavelet Tranform.
  WARNING: There is no reconstruction factor

  Parameters:
    X : 2d array_like object
      transformed data
    dt : float
      time step
    scales : 1d array_like object
      scales
    wf : string ('morlet')
      wavelet function
    w0 : float
      wavelet function parameter

  Returns:
    x : 1d numpy array
      data
  """  
    
  X_arr = np.asarray(X)
  scales_arr = np.asarray(scales)

  if X_arr.shape[0] != scales_arr.shape[0]:
    raise ValueError('X, scales: shape mismatch')

  X_ARR = np.empty_like(X_arr)
  for i in range(scales_arr.shape[0]):
    X_ARR[i] = X_arr[i] / np.sqrt(scales_arr[i])
  
  x = np.sum(np.real(X_ARR), axis=0)
 
  return x
