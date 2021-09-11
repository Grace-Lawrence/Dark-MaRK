#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
from __future__ import print_function, absolute_import, division
import astropy.units as u
import numpy as np
import astropy.constants as const
import scipy.integrate as sci
from scipy import special 

from mermaid.velocity_int import gaussian, vint
from mermaid.nat_unit import to_natural



#Define Globals 
GeVc2 = u.def_unit('GeVc2', u.GeV/(const.c)**2)
tru = 1./(u.d*u.kg)

__all__ = ["vmin_func", "E_0", "R_0", "r_kin", "qf","helm_form", "Form_Factor", 
           "reduced_mass", "reduced_mass_ratio", "k_ratio", "k_func"]

def vmin_func(nib, rkin, E_r):
    """Calculates the minimum detectable velocity for a dm particle of certain recoil energy"""
    value = (np.sqrt((2.*E_r)/(rkin*nib._dm.mass))).to(u.km/u.s)
    return(value)

def E_0(nib):
    """Average energy of incoming dark matter particles """
    return (0.5*nib._dm.mass*nib._vdf.v_0**2.).to(u.keV)

def R_0(nib):
    """Calculates R0, the fly-through rate in tru units"""
    value = ((2./np.sqrt(np.pi)) * (const.N_A/(nib._detector.atomic_number*(u.g/u.mol))) * ((nib._dm.density/nib._dm.mass) *nib._dm.cross_section* (nib._vdf.v_0))).to(tru)
    return(value)

def r_kin(nib):
    """Unitless Kinematic Factor"""
    return (4.*nib._dm.mass*nib._detector.M_T)/((nib._dm.mass+nib._detector.M_T)**2.)

def qf(nib, e_val):
    """Defines a scalar quenching factor for sodium and iodine"""
    if nib._det_var == 'Na':
        qf = 0.3 
    elif nib._det_var == 'Iod':
        qf = 0.09
    else:
        qf = 0.3
    return np.ones(len(e_val))*qf

def helm_form(nib):
    """ Calculate the Helm Form Factor using the Fermi Distribution """
    a = 0.52 *u.fm #Constants taken from Shield's thesis after equation 8.19 (http://inspirehep.net/record/1466707/files/Shields-Thesis.pdf)
    s = 0.9 *u.fm
    c = ((1.23*nib._detector.atomic_number**(1./3.))-0.6)*u.fm
    hbar = (197.3*(u.MeV*u.fm)).to(u.eV*u.fm)

    r_n_sq = (c**2.)+((7./3.*np.pi**2.*a**2.))-(5.*s**2.) #Eq 4.11 from Lewin and Smith
    r_n = np.sqrt(r_n_sq)
    E_r_n = nib._dm.Er_range
    q = (np.sqrt(((2.*(0.932)*GeVc2)*(nib._detector.atomic_number)*(E_r_n.to(u.GeV))))/hbar*const.c)
    print(f'NEW Q: {q[0]}')
    q = q.to(1/u.fm)
    print(f'Og q values: {q.min()}, {q.max()} ')
    qrn = (q*r_n)
    qs = q*s
    F_qH = 3.*np.exp((-1.*qs**2.)/2.)*((np.sin(qrn.value)-qrn.value*np.cos(qrn.value))/(qrn.value)**3.) 
    return F_qH, q, qrn, E_r_n.to(u.keV)

def Form_Factor(nib, LF=False, CERN=True):
    """Generate a new descriptor for this"""
    if CERN or LF:
        Q_energy = nib._dm.Er_range.to(u.GeV)
        hbarc = (const.hbar * const.c).to(u.GeV*u.fm) #this is (hbar * c)
        q_mtm = (np.sqrt(2*to_natural(nib._detector.M_T.to(u.kg))*to_natural(Q_energy)))
        #To natural where Energy and Mtm are now GeV, and GeV becomes 1/fm by dividing by (hbar*c)
        q_mtm /= hbarc
        q_mtm = q_mtm.to(1/u.fm)
        if LF:
            r = (1e-13 * u.cm * (0.3 + 0.91 * (to_natural(nib._detector.M_T.to(u.kg)).to(u.GeV).value)**(1/3))).to(u.fm)
            # Lewin&Smith86 use r = (0.3 + 0.89 * (MT.to(u.u).value)**(1/3))*u.fm 
            s = 1.0 * u.fm # this is approximate, and is given as 0.9*u.fm in Lewin and Smith
        elif CERN:
            r = (1.2*u.fm) * (nib._detector.M_T.to(u.u).value)**(1/3) # Using Atomic Number value
            s = 1.0 * u.fm # this is approximate          
        r1 = (np.sqrt(r**2-5*s**2)).to(u.fm)
        qr1 = (q_mtm*r1).value
        qs = (q_mtm*s).value
        j1 = special.spherical_jn(1,qr1) # use scipy for full accuracy
        
        # LF explicity use the Sine and Cosine components of j1
        #F_Q = ((3*(np.sin(qr1)-qr1*np.cos(qr1)))/( (qr1)**3)) * np.exp(-(qs)**2 / 2.)
        #F_Q = (3*j1/qr1) * np.exp(-(qs)**2 / 2.) 

        #This is the Wood-Saxon form from CERN paper but note their eqn 7.33 is written as F(Q) but it's actually F^2(Q) already
        #ie. F_Q_sq = (3*j1/qr1)**2. * np.exp(-(qs)**2) 
        # Eqn 9 in https://arxiv.org/pdf/hep-ph/0608035.pdf makes clear that the below is correct
        F_Q = (3*j1/qr1) * np.exp(-(qs)**2/ 2.)     
    else:
        F_Q = np.ones(len(nib._dm.Er_range))
    return F_Q, qr1
    
def reduced_mass(a,b):
    """Calculates the reduced mass for mass a and mass b"""
    red_m = (a*b)/(a+b)
    return red_m

def reduced_mass_ratio(dm_mass,target_mass,proton_mass):
    """Caluclate the reduced mass ratio of dm mass-target mass/dm mass-proton mass"""
    num = reduced_mass(dm_mass, target_mass)
    num = num.to(GeVc2)
    denum = reduced_mass(dm_mass, proton_mass)
    denum = denum.to(GeVc2)
    return(num/denum)

def k_ratio(nib, ratio):
    """ Calculates k0 and k1 returning their ratio per Lewin and Smith (after eq 2.2) """
    x = nib._dm.Er_range.value
    y = gaussian(x,nib)
    k0 = k_func(1000000, 150,x,y,nib)
    k1 = k_func(nib._vdf.v_max.value,150,x,y, nib)
    if ratio == True:
        return k1/k0 # ~0.9965
    else: 
        return k1

def k_func(v_max, day, x, y,nib):
    """Calculates normalization constant, k"""
    k, err = sci.quad(vint, 0, abs(v_max), args=(0.,2.,nib))        
    k *= 2*np.pi
    return k
