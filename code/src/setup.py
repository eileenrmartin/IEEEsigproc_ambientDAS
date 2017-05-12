from distutils.core import setup, Extension


corrs_module = Extension('corrs_module', sources=['corrs_module.cpp'])
crosscorr_module = Extension('crosscorr_module', include_dirs = ['/usr/local/tbb/include'], library_dirs = ['/usr/local/tbb'], libraries = ['tbb','rt'], sources=['crosscorr_module.cpp'])


import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

setup(ext_modules=[crosscorr_module,corrs_module])
