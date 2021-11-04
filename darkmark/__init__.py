#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:30:28 2019

@author: gracelawrence
"""
from .__version__ import version as __version__
from . import _globals
from ._globals import *
from . import annual_mod_fits
from . import simload
from . import framewrap
from . import sampling_transformations
from . import dm_detector_class
from . import nat_unit
from . import velocity_int

__name__ = "Dark Matter Rate Kalculator (Dark MaRK)"
__author__ = "Grace Lawrence (@Grace-Lawrence)"
__all__ = ['annual_mod_fits, simload', 'profiles', 'framewrap',
           'sampling_transformations','dm_detector_class',
            'nat_unit', 'velocity_int']
__all__.extend(_globals.__all__)
