#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
import astropy.units as u
import numpy as np
import scipy.integrate as sci

from darkmark.velocity_int import thint


__all__ = ["spectral_func"]


def __vterm__(nib):
    """Calculates the average velocity term <v> following mathematics in the cell below."""
    v_term = (__vterm_numerator__(nib._vdf.v_max.value,nib))/(__vterm_denominator__(nib._vdf.v_max.value,nib))
    return v_term*u.km/u.s

def __vterm_numerator__(v_max,nib):
    """Integrates velocity function between 0 and vmax"""
    k, err = sci.quad(__vint_numerator__, 0, abs(v_max), args=(0.,2.,nib))        
    k *= 2*np.pi
    return k

def __vterm_denominator__(v_max, nib):
    """Integrates velocity function between 0 and vmax"""
    k, err = sci.quad(__vint_denominator__, 0, abs(v_max), args=(0.,2.,nib))        
    k *= 2*np.pi
    return k

def __vint_numerator__(v,vE,ind,nib):
    """Continues velocity integration returning the numerator from the <v> expression in cell below"""
    ans, err = sci.quad(thint, 0., np.pi, args=(v,vE,nib))
    return(v**3*ans)

def __vint_denominator__(v,vE,ind,nib):
    """Continues velocity integration returning the denominator from the <v> expression in cell below"""
    ans, err = sci.quad(thint, 0., np.pi, args=(v,0,nib))
    return(v**2*ans)

def gaussian(u,nib):
    """Defines a Gaussian Boltzmann Distribution"""
    return(np.exp(-(u**2./(nib._vdf.v_0.value**2.))))

def __num_int__(nib, ind, v_array):
    sum_vel = np.sum(v_array)
    return((1/len(v_array))*sum_vel)
    
def __num_vterm__(nib, v_array):
    return(__num_int__(nib, 1, v_array))#/__num_int__(nib, 2, v_array))


def spectral_func(nib, F_qH, day, vE,vmin,speed,norm_const,cf_ls= True):
    """Sums over the particle velocities to calculate the differential count rate"""
    rate = np.empty([len(nib._dm.Er_range.value)])
    vmax = nib._vdf.v_max.value 
    speed_vmax = speed<(vmax+vE.value)
    vmin_t_arr = np.apply_along_axis(lambda x: vmin[x].value, 0, range(len(nib._dm.Er_range)))#Boolean array defining which velocity values are < vmax+vE
    
    #For each energy bin and it's corresponding minimum velocity
    for index, vmin_t in enumerate(vmin_t_arr): #Looping through energy bins and minimim velocities
        speed_cut = speed[(speed > vmin_t)*speed_vmax] #Perform the velocity cut
        #Perform Summation
        if len(speed_cut) == 0:
            rate[index]=0.
        else:
            rate[index] = 1./len(speed)* np.sum((1.)/(speed_cut)) #perform the integration/summation
    
    #Normalise
    if cf_ls:
        rate_norm =(rate*u.s/u.km)*F_qH**2*norm_const #apply the constants and form factor terms
    return rate_norm
