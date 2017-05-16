from distutils.core import setup, Extension


corrs_module = Extension('corrs_module', sources=['corrs_module.cpp'])
onebitcrosscorr_module = Extension('onebitcrosscorr_module', include_dirs = ['/usr/local/tbb/include'], library_dirs = ['/usr/local/tbb'], libraries = ['tbb','rt'], sources=['onebitcrosscorr_module.cpp'])



import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

setup(ext_modules=[onebitcrosscorr_module,corrs_module])
