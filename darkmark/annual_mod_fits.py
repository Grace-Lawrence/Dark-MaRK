#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:52:42 2020

@author: gracelawrence
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import unicodedata

__all__ = ["fit_curve"]

def fit_func_S(t, S_0, S_m,t0):
    """
    Functional form for annual modulation from Bernabei et al. (2008) with 
    seperate S0, Sm terms

    Parameters
    ----------
    t : Float
        Period over which to perform the fit (in days).
    S_0 : Float
        The constant part of the signal.
    S_m : Float
        THe modulation amplitude.
    t0 : Float
        Peak day (phase).
        
    Returns
    -------
    The fitted value for the signal as defined by Eq 1 in Eur.Phys.J.C56:333-355,2008

    """
    return(S_0 + S_m*np.cos(((2*np.pi)/365)*(t-t0)))

def fit_func_A(t,A,t0):
    """
    Functional form for annual modulation

    Parameters
    ----------
    t : Float
        Period over which to perform the fit (in days).
    A : Float
        Modulation amplitude.
    t0 : Float
        Peak day (phase).

    Returns
    -------
    The array of fit values for the given period, t, for the residual count signal.

    """
    return(A*np.cos(((2*np.pi)/365)*(t-t0)))


def fit_curve(nib, y_data, fit, verbose = False):
    """
    Using scipy.optimize.curve_fit to find fit parameters for annual 
    modulation curves

    Parameters
    ----------
    y_data : Array of floats
        The annual modulation curve data in dru units.
    fit : String
        Defines whether the function will fit the full signal, or residual counts.
    verbose : String, optional
        DESCRIPTION. The default is False. If True, the function will print the 
        fit paramters when it is called. The function will also generate a plot
        showing the original signal, with the overlaid fit.

    Returns
    -------
    params : Array of floats
        The fit parameters for the input signal (ydata).
    params_covariance : Array of floats
        The associated errors for params.

    """
    x_data = np.linspace(0.001,nib._vdf.calc_day,nib._vdf.calc_day,dtype=int)
    if fit == 'A': #Subtracting the median from data to examine the residuals
        y_data = np.subtract(y_data,np.mean(y_data))
        params, params_covariance = optimize.curve_fit(fit_func_A, x_data, y_data, 
                                                       p0 = [0.023,152.5]) 
        if verbose:
            print(f'A: Modulation Amplitude ={params[0]},\nt0: The phase = {params[1]}')
            plt.clf()
            plt.figure(figsize=(6, 4))
            plt.scatter(x_data, y_data, label='Data')
            plt.plot(x_data, fit_func_A(x_data, params[0], params[1]),
                     label='Fitted function', color='red')
            plt.xlabel('Time (d)')
            plt.ylabel(r'$ \frac{dR}{dE_r} (dru)$')
            plt.title('Residual Count Rate')
            plt.legend()
            plt.show()
    elif fit == 'S': #Fitting both the constant rate S0 and the modulation S_0
        max_day = np.where(y_data == y_data.max())[0]
        params, params_covariance = optimize.curve_fit(fit_func_S, x_data, y_data, 
                                                       p0 = [1,1,max_day], 
                                                       bounds=([-5,-5,0], [5., 5., 365])) 
        perr = np.sqrt(np.diag(params_covariance))
        if verbose:
            print(f"S_0: Constant part of the signal = {params[0]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[0]}") 
            print(f"S_m: Modulation Amplitde = {params[1]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[1]}") 
            print(f"t_0: The phase = {params[2]} " + unicodedata.lookup("Plus-Minus Sign") +f" {perr[2]}") 
    return params, params_covariance

