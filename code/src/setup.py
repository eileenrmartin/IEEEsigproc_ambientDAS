from distutils.core import setup, Extension
import numpy

module1 = Extension('corrs', sources=['corrsmodule.cpp'])

import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

setup (name = 'PackageName', version = '1.0', description = 'This is a demo package', ext_modules = [module1])
