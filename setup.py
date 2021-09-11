# -*- coding: utf-8 -*-

"""
Setup file for the Dark MaRK package.
"""

# %% IMPORTS
# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Get the extra requirements list
with open('requirements_extras.txt', 'r') as f:
    requirements_extras = f.read().splitlines()

# Read the __version__.py file
with open('darkmark/__version__.py', 'r') as f:
    vf = f.read()

# Obtain version from read-in __version__.py file
version = re.search(r"^_*version_* = ['\"]([^'\"]*)['\"]", vf, re.M).group(1)

# Setup function declaration
setup(name="darkmark",
      version=version,
      author="Grace Lawrence",
      author_email='glawrence@swin.edu.au',
      description=(""),
      long_description=long_description,
      url="",
      project_urls={},
      license='BSD-3',
      platforms=['Windows', 'Mac OS-X', 'Linux', 'Unix'],
      classifiers=[],
      keywords=(""),
      python_requires='>=3.6, <4',
      packages=find_packages(),
      package_dir={'darkmark': 'darkmark'},
      include_package_data=True,
      install_requires=requirements,
      extras_require={'extras': requirements_extras},
      zip_safe=False,
      )
