#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:46:57 2019

@author: gracelawrence
"""
import astropy.units as u
import astropy.constants as const

# All declaration
__all__ = ['show_plot','dru','tru','pb','GeVc2','keVc2']


# %% GLOBAL DEFINITIONS
show_plot = False
dru = 1./(u.d*u.kg*u.keV)
tru = 1./(u.d*u.kg)
pb = 1e-36
GeVc2 = u.def_unit('GeVc2', u.GeV/(const.c)**2)
keVc2 = u.def_unit('keVc2', u.keV/(const.c)**2)
