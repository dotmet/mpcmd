import sys
import os
##from pybind11 import get_cmake_dir
##from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.

#ext_modules = [
#    Pybind11Extension("cpp",
#        [os.getcwd()+"/*.cpp"],
#        # Example: passing in the version to the compiled code
#        define_macros = [('VERSION_INFO', __version__)],
#        ),
#]

setup(
    name = "PyMPCD_MD",
    version = __version__,
    author = "Even Wong",
    author_email = "evenwong@stu.cdut.edu.cn",
    url = "https://github.com/Gddr100x/PyMPCD-MD",
    description = "A python-based simulation package about mpcd and md.",
    long_description = "",
    extras_require = {},
##    ext_modules=ext_modules,
    packages = find_packages(),
    zip_safe = False,
    python_requires = ">=3.0",
    install_requires = [
        'matplotlib',
        'numpy',
        'scipy',
        'numba',
        'gsd',
        'memory_profiler',
        'seaborn',
        ],
)
