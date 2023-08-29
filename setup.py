import sys
import os
from setuptools import setup, find_packages

__version__ = "0.0.6"

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = "mpcmd",
    author = "Even Wong",
    version = __version__,
    author_email = "wdyphy@zju.edu.cn",
    url = "https://github.com/dotmet/mpcmd",
    description = "A python-based simulation package using mpcd and md.",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
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
        'scikit-learn',
        ],
)
