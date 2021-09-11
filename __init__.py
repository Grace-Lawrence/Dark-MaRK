#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:30:28 2019

@author: gracelawrence
"""
from __future__ import absolute_import, print_function, division

from .__version__ import version as __version__
from . import _globals
from ._globals import *
from . import annual_mod_fits
from . import simload
from . import profiles
from . import framewrap
from . import skyplot
from . import sampling_transformations
from . import dm_detector_class
from . import calc_rate_funcs
from . import test
from . import nat_unit
from . import velocity_int

__name__ = "Dark MAtter Rate Kalculator (Dark Mark)"
__author__ = "Grace Lawrence (@Grace-Lawrence)"
__date__ = '2019-03-25'
__cite__ = 'https://github.com/Grace-Lawrence/DarkMark'
__all__ = ['annual_mod_fits, simload', 'profiles', 'framewrap', 'skyplot',
           'sampling_transformations','dm_detector_class', 'calc_rate_funcs',
           'test', 'nat_unit, velocity_int']
__all__.extend(_globals.__all__)
