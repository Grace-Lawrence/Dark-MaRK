#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from darkmark.annual_mod_fits import fit_curve
from darkmark import dru, tru

__all__ = [ "annual_mod", "find_amplitude"]


def Qf_Ge(nib,e_val):
    """
    Returns Quenching Factor Values for Germanium.

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    e_val : Float, or array of floats
        Energy at which to evaluate.

    Returns
    -------
    Q : Float, or array of floats
        Quenching Factor.

    """
    k = 0.1789
    Q = (k*__g_e__(e_val))/(1+(k*__g_e__(e_val)))
    
    return Q

def __g_e__(e_val):
    return((3*__eps_Ge__(e_val)**(0.15))+(0.7*__eps_Ge__(e_val)**(0.6))+(__eps_Ge__(e_val)))

def __eps_Ge__(e_val):
    Z = 32
    Z_term = 11.5*(Z**(-7/3))
    return(np.multiply(Z_term, e_val.value))


def qf(nib, e_val):
    """
    Defines a scalar quenching factor for sodium and iodine

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    e_val : Float, or array of floats
        Energy at which to evaluate.

    Returns
    -------
    Q : Float, or array of floats
        Quenching Factor.

    """
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
    """
    Integrates the rate over a EVee (quenched) energy range to generate an annual modulation array

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    rate : Array of floats
        Spectral rate function arrays to integrate over.
    min_ee : Float
        The lower energy limit of the integral.
    max_ee : Float
        The upper energy limit of the integral.
    recoil : String
        Option of 'Nuclear' or 'E_Equiv', to integrate over recoil eneergies (keV)
        or quenched energy range in Electron Equivalent energies (keVee).

    Returns
    -------
    E_r_array : Array of floats
        Returns the annual modulation values in dru units, for the specified period.

    """
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
    """
    Calculates the amplitude value for the annual modulation array.
    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    am_array : Array of floats.
        Spectral rate function arrays to integrate over.

    Returns
    -------
    amp : Float
         Amplitude of the annual modulation signal.

    """

    params, params_covariance = fit_curve(nib, am_array, 'A')
    amp = params[1]
    return amp
