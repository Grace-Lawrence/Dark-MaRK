#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
import astropy.units as u
import numpy as np
import astropy.constants as const
from scipy import special 


from darkmark.nat_unit import to_natural
from darkmark import tru, GeVc2 



__all__ = ["vmin_func", "E_0", "R_0", "r_kin", "qf","Form_Factor", 
           "reduced_mass", "reduced_mass_ratio"]

def vmin_func(nib, rkin, E_r):
    """
    The minimum detectable velocity for a dark matter particle of certain recoil energy

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    rkin : Float
        The kinematic factor, dependent on M_T, M_D.
    E_r : Float
        Recoil energy at which to evaluate vmin.

    Returns
    -------
    Float
        v_min
    """
    value = (np.sqrt((2.*E_r)/(rkin*nib._dm.mass))).to(u.km/u.s)
    return(value)

def E_0(nib):
    """
    Average energy of incoming dark matter particles

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.

    Returns
    -------
    Float
        E_0

    """
    return (0.5*nib._dm.mass*nib._vdf.v_0**2.).to(u.keV)

def R_0(nib):
    """
    Calculates R0, the fly-through rate in tru units

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.

    Returns
    -------
    R0

    """
    value = ((2./np.sqrt(np.pi)) * (const.N_A/(nib._detector.atomic_number*(u.g/u.mol))) * ((nib._dm.density/nib._dm.mass) *nib._dm.cross_section* (nib._vdf.v_0))).to(tru)
    return(value)

def r_kin(nib):
    """
    Unitless Kinematic Factor

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.

    Returns
    -------
    r_kin

    """
    return (4.*nib._dm.mass*nib._detector.M_T)/((nib._dm.mass+nib._detector.M_T)**2.)

def qf(nib, e_val):
    """
    Defines a scalar quenching factor for sodium and iodine

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    e_val : Float
        Energy at which to evaluate the quenching factor.

    Returns
    -------
    Quenching Factor

    """
    if nib._det_var == 'Na':
        qf = 0.3 
    elif nib._det_var == 'Iod':
        qf = 0.09
    else:
        qf = 0.3
    return np.ones(len(e_val))*qf



def Form_Factor(nib, LF=False, CERN=True):
    """
    Returns a form factor which accounts for the fact that simple scattering is not an appropriate way to model the interaction of large target nuclei with heavy WIMP dark matter, a model for nuclear charge density is introduced into dark matter detection rate calculations.
    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    LF : Boolean, optional
        Option for different assumption in the Form Factor equation pertaining to
        r (effective nuclear radius) and s (nculear skin thickness). It follows
        paper,Phys.Rev.D70:043501,2004 . The default is False.
    CERN : Boolean, optional
        Option for different assumption in the Form Factor equation pertaining to
        r (effective nuclear radius) and s (nculear skin thickness).  It follows
        paper, Phys.Rept. 267 (1996) 195-373. The default is True.

    Returns
    -------
    F_Q : Array of floats
        Form factor for corresponding recoil energy.
    qr1 : Array of floats
        Momentum term, dependent on energy.

    """
    if CERN or LF:
        Q_energy = nib._dm.Er_range.to(u.GeV)
        hbarc = (const.hbar * const.c).to(u.GeV*u.fm) #this is (hbar * c)
        q_mtm = (np.sqrt(2*to_natural(nib._detector.M_T.to(u.kg))*to_natural(Q_energy)))
        q_mtm /= hbarc #To natural where Energy and Mtm are now GeV, and GeV becomes 1/fm by dividing by (hbar*c)
        q_mtm = q_mtm.to(1/u.fm)
        if LF:
            r = (1e-13 * u.cm * (0.3 + 0.91 * (to_natural(nib._detector.M_T.to(u.kg)).to(u.GeV).value)**(1/3))).to(u.fm)
            s = 1.0 * u.fm # this is approximate, and is given as 0.9*u.fm in other literature.
        elif CERN:
            r = (1.2*u.fm) * (nib._detector.M_T.to(u.u).value)**(1/3) # Using Atomic Number value
            s = 1.0 * u.fm # this is approximate          
        r1 = (np.sqrt(r**2-5*s**2)).to(u.fm)
        qr1 = (q_mtm*r1).value
        qs = (q_mtm*s).value
        j1 = special.spherical_jn(1,qr1) # use scipy for full accuracy
        
        F_Q = (3*j1/qr1) * np.exp(-(qs)**2/ 2.)     
    else:
        F_Q = np.ones(len(nib._dm.Er_range))
    return F_Q, qr1
    
def reduced_mass(a,b):
    """
    Calculates the reduced mass for mass a and mass b

    Parameters
    ----------
    a : Float
        Mass value.
    b : Float
        Mass value.

    Returns
    -------
    red_m : Float
        Reduced mass of masses a and b. 

    """
    red_m = (a*b)/(a+b)
    return red_m

def reduced_mass_ratio(dm_mass,target_mass,proton_mass):
    """
    Caluclate the reduced mass ratio of dm mass-target mass/dm mass-proton mass

    Parameters
    ----------
    dm_mass : Float
        Mass of dark matter - needs to have units given.
    target_mass : Float
        Mass of target detector nuclei - needs to have units given.
    proton_mass : Float
        Mass of a proton.

    Returns
    -------
    The reduced mass ratio of the M_D,M_T reduced mass with the M_D,M_p reduced
    mass.

    """
    num = reduced_mass(dm_mass, target_mass)
    num = num.to(GeVc2)
    denum = reduced_mass(dm_mass, proton_mass)
    denum = denum.to(GeVc2)
    return(num/denum)

