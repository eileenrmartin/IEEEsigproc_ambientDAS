from distutils.core import setup, Extension
#import numpy

corrs_module = Extension('corrs_module', sources=['corrs_module.cpp'])

import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

setup(ext_modules=[corrs_module])
