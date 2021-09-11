#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
from __future__ import print_function, absolute_import, division
import astropy.units as u
import numpy as np
import scipy.integrate as sci
from scipy.interpolate import InterpolatedUnivariateSpline
from mermaid.annual_mod_fits import fit_curve

__all__ = ["vint", "vint_ang", "thint", "gaussian", "annual_mod", "find_amplitude"]

def vint(v,vE,ind,nib):
    """Integrates a functional velocity distribution"""
    ans, err = sci.quad(thint, 0., np.pi, args=(v,vE,nib))
    return (v**(ind)*ans)

def vint_ang(v,vE,ind,nib):
    """Integrates a functional velocity distribution with angular dependence"""
    cos_theta_min = (nib._vdf.v_max.value**2.-v**2.-vE**2.)/(2.*v*vE)
    theta_min = np.arccos(cos_theta_min)
    ans, err = sci.quad(thint,theta_min, np.pi ,args=(v,vE,nib))
    return (v**(ind)*ans)

def thint(th, v, vE, nib):
    """Defines a Functional Form for the velocity distribution function"""
    u = np.sqrt(v**2.+vE**2.+(2.*v*vE*np.cos(th)))
    if nib._vdf_var == 'SHM' or nib._K_form:
        return gaussian(u, nib)*np.sin(th)
    
def gaussian(u,nib):
    """Defines a Gaussian Boltzmann Distribution"""
    return(np.exp(-(u**2./(nib._vdf.v_0.value**2.))))

def Qf_Ge(nib,e_val):
    "Returns Quenching Factor Values for Germanium"
    k = 0.1789
    Q = (k*g_e(e_val))/(1+(k*g_e(e_val)))
    return Q

def g_e(e_val):
    return((3*eps_Ge(e_val)**(0.15))+(0.7*eps_Ge(e_val)**(0.6))+(eps_Ge(e_val)))

def eps_Ge(e_val):
    Z = 32
    Z_term = 11.5*(Z**(-7/3))
    return(np.multiply(Z_term, e_val.value))


def qf(nib, e_val):
    """Defines a scalar quenching factor for sodium and iodine"""
    if nib._det_var == 'Na':
        qf = 0.3 
    elif nib._det_var == 'Iod':
        qf = 0.09
    elif nib._det_var == 'Ge':
        qf = Qf_Ge(nib,e_val)
        return qf
    else:
        qf = 0.3
    return np.ones(len(e_val))*qf

def annual_mod(nib, rate, min_ee, max_ee, recoil):
    """Integrates the rate over a EVee (quenched) energy range to generate an annual modulation array"""
    E_r_array = np.zeros(len(nib._vdf.calc_day))
    
    #Calculate the quenching factors depending on if the recoils are nuclear or electron-equivalent
    if recoil == "E_Equiv":
        quench_fac = qf(nib, nib._dm.Er_range) #nib._dm.Er_range -> Nuclear Recoil Energy
        E_v_ee = nib._dm.Er_range*quench_fac
        E_v_nr_new = InterpolatedUnivariateSpline(E_v_ee,nib._dm.Er_range)
        min_nr = E_v_nr_new(min_ee)*u.keV
        max_nr = E_v_nr_new(max_ee)*u.keV
    elif recoil == "Nuclear":
        quench_fac = np.ones(len(nib._dm.Er_range))
        min_nr = min_ee
        max_nr = max_ee

    #Interpolate the rate over the energy region-of-interest, and integrate
    for i in range(0, len(nib._vdf.calc_day)): #for each day of the year
        rate_spline = InterpolatedUnivariateSpline(nib._dm.Er_range, rate[i,:]) 
        if min_ee == max_ee:
            if recoil == 'Nuclear':
                E_r_array[i] = rate_spline(min_nr.value)
            elif recoil == 'E_Equiv':
                dER_dEee = E_v_nr_new.__call__(min_ee.value,1) 
                E_r_array[i] = rate_spline(min_nr.value) * dER_dEee
        else: 
            E_r_array[i] = rate_spline.integral(min_nr.value, max_nr.value) 
    
    #Normalize over the energy range to get dru units
    if min_ee == max_ee:
        E_r_array = (np.array(E_r_array)*tru/((1*u.keV))).to(dru)
    else:
        if recoil == 'E_Equiv':
            E_r_array = (np.array(E_r_array)*tru/((max_ee-min_ee))).to(dru)
        elif recoil == 'Nuclear':
            E_r_array = (np.array(E_r_array)*tru/((max_nr-min_nr))).to(dru)
    return E_r_array


def find_amplitude(nib, am_array, fit):
    if fit:
        params, params_covariance = fit_curve(am_array, 'A')
        amp = params[1]
    else:
        amp = abs(np.median(am_array)-np.max(am_array))
    return amp
