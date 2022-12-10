import sys
import os
from setuptools import setup, find_packages

__version__ = "0.0.1"

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = "PyMPCD_MD",
    version = __version__,
    author = "Even Wong",
    author_email = "evenwong@stu.cdut.edu.cn",
    url = "https://github.com/Gddr100x/PyMPCD-MD",
    description = "A python-based simulation package about mpcd and md.",
    long_description = "A python-based simulation package about mpcd and md.",
    extras_require = {},
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
