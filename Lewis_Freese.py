#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 21:45:48 2020

@author: gracelawrence
"""
from __future__ import print_function, absolute_import, division
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import special
from mermaid.freq_eq import r_kin, R_0, E_0
from mermaid.dm_detector_class import Nibbler, int_func, calc_real_vE
from mermaid.nat_unit import to_natural


__all__ = ['Form_Factor']


#Define Globals 
GeVc2 = u.def_unit('GeVc2', u.GeV/(const.c)**2)
keVc2 = u.def_unit('keVc2', u.keV/(const.c)**2)
dru = 1./(u.d*u.kg*u.keV)
tru = 1./(u.d*u.kg)
pb = 1e-36
solar_x = 232.24
solar_y = 11.1
solar_z = 7.25
Nsteps = 10000 #Number of velocity samples

def reduced_m(M_x,M_n):
    a = (M_x*M_n/(M_x+M_n))
    return a

def Lewis_Freese_Form_Factor(nib):
    M = 73*GeVc2
    E = nib._dm.Er_range.to(u.GeV)
    hbar = (197.3*(u.MeV*u.fm)).to(u.eV*u.fm)
    Q = (np.sqrt(2*M*E))/hbar*const.c
    Q = Q.to(1/u.fm)
    r = (((0.91*(73)**(1/3)+0.3)*10**(-13))*u.cm).to(u.fm)
    s = 1*u.fm
    r1 = ((r**2-5*s**2)**(1/2)).to(u.fm)
    Qr1 = (Q*r1)
    F_Q = ((3*(np.sin(Qr1.value)-Qr1.value*np.cos(Qr1.value)))/(Qr1.value**3))*(np.exp(-(Q.value**2*s.value**2)/2))
    return np.array(F_Q), Qr1





def v_min_freese(nib):
    mr = reduced_m(nib._dm.mass,(nib._detector.M_T))
    vmin = (np.sqrt((nib._detector.M_T*nib._dm.Er_range)/(2*mr**2))).to(u.km/u.s)
    return vmin

def vmin_freese(MDM,MT,Erange):
    mr = redmass(MDM,MT)
    vmin = (np.sqrt((MT*Erange)/(2.*mr**2))).to(u.km/u.s) #vmin in LF 2004 has m_N, it means the target nucleus mass
    return vmin

def erf_calc(nib,n_day):
    """Calculates the semi-analytical form of the differential energy equation"""
    v_es = 29.8*u.km/u.s
    v_sh = 233*u.km/u.s
    t = (nib._vdf.calc_day)/len(nib._vdf.calc_day) #time in years
    phi_h = 2.61
    sigma_h = 270*u.km/u.s
    v_eh = calc_real_vE(nib._vdf.calc_day, 'Astropy')#(np.sqrt(v_es**2+v_sh**2+(2*v_es*v_sh*np.cos(2*np.pi*t-phi_h)))).to(u.km/u.s)
    vmin_array = v_min_freese(nib).to(u.km/u.s) 

    if n_day == 1:
        print(f'Earth Velocity Example: {v_eh[0]}')
        plt.clf()
        plt.plot(v_eh)
        plt.title('Earth Velocity')
        plt.xlabel('Time (days)')
        plt.ylabel('vE')
        plt.show()
        plt.clf()
        plt.plot(nib._dm.Er_range, vmin_array)
        plt.xlabel('Vmin')
        plt.ylabel('Time (days)')
        plt.title('Vmin')
        plt.show()
    
    mr = reduced_m(nib._dm.mass,(nib._detector.M_T))
    dru_array = np.empty(len(nib._dm.Er_range))
    coefficent = ((nib._dm.cross_section*nib._dm.density)/(2*mr**2*nib._dm.mass)).to(u.km/(u.d*u.keV*u.kg*u.s))
    if n_day == 1:
        print(f'Coefficient: {coefficent}')
    for index, energy in enumerate(nib._dm.Er_range):
        vmin = vmin_array[index]
        dru_array[index] = (1/(2*(v_eh[n_day].to(u.km/u.s).value))) * (sc.special.erf((np.sqrt(2)*(vmin.to(u.km/u.s).value+v_eh[n_day].to(u.km/u.s).value))/(np.sqrt(3)*sigma_h.to(u.km/u.s).value))-sc.special.erf((np.sqrt(2)*(vmin.to(u.km/u.s).value-v_eh[n_day].to(u.km/u.s).value))/(np.sqrt(3)*sigma_h.to(u.km/u.s).value)))

    dru_array = coefficent * (dru_array*u.s/u.km)
    if n_day ==1:
        print(f'Spectral Rate Function at Day 0 \n min:{dru_array.min()} \n max: {dru_array.max()}')
    return dru_array.to(dru).value

def ERF_Lewis_Freese(detector, DM_model, Halo_model, sigma_cs, v0,rho_dm, M_dm, sigma_0, v_max, period,min_ee, max_ee):
    """Main Function Call """
    #Define Objects and Classes
    detvar=(detector)
    dmvar=(DM_model)
    vdf_var =(Halo_model)
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max, period)
    print(f'Detector \n Name: {nib._detector.name},\n Atomic Mass: {nib._detector.atomic_mass},\n Reduced Mass: {nib._detector.M_T }, \n Mass Number: {nib._detector.atomic_number }')
    print(f'Dark Matter Candidate \n Density: {nib._dm.density},\n Mass: {nib._dm.mass},\n Cross Section 0: {nib._dm.cross_section},\n Energy Range: {nib._dm.Er_range.min().value}-{nib._dm.Er_range.max()} ')
    print(f'Velocity Distribution \n Esc Velocity: {nib._vdf.v_max},\n Time Period: {nib._vdf.calc_day.max()} days,\n Median DM Velocity: {nib._vdf.v_0}')
    print(f'Interaction Func = {int_func(nib._detector.atomic_mass.value)}')
    # F_qH_Smith, q, qrn, Er_n = helm_form(nib)
    # F_qH_Freese, Qr1 = Lewis_Freese_Form_Factor(nib)
    F_qH, qr1_al= Form_Factor_Alan(nib, LF=False, CERN=True)
    Form_Fac = F_qH**2.
    # plt.clf()
    # plt.plot(Qr1,F_qH_Smith**2, label = 'Form_Fac_Smith')
    # plt.plot(q,F_qH_Freese**2, label = 'Form_Fac_Freese')
    # plt.plot(qr1_al, F_qH_Alan**2, label = 'Alan Freese')
    # plt.xlabel('q')
    # plt.ylabel('FqH**2')
    # plt.title('Form Factor')
    # plt.yscale('log')
    # plt.legend()
    # plt.show()
    rate_array = np.empty([len(nib._vdf.calc_day), len(nib._dm.Er_range)])
    for i, day in enumerate(nib._vdf.calc_day):
        # print(f'day: {day}')
        rate_array[i] = erf_calc(nib,i)*Form_Fac
    rate_array = np.array(rate_array)*dru
    R0 = R_0(nib)
    E0 = E_0(nib)
    rkin = r_kin(nib)
    plt.clf()
    plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*rate_array[0,:],s=1, label = 'Summer')
    plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*rate_array[90,:],s=1, label = 'Autumn')
    plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*rate_array[180,:],s=1, label = 'Winter')
    plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*rate_array[270,:],s=1, label = 'Spring')
    plt.xlabel(r'$ \frac{E}{E_0r}$')
    plt.ylabel(r'$ \frac{E_0r}{R_0}\frac{dR}{dE_r}$')
    plt.xlim(0,10)
    plt.title(f'Seasonal Variation of Rate Spectrum - {nib._detector.name}')
    plt.legend()
    plt.show()
    # rate = annual_mod(nib, rate_array, min_ee, max_ee, recoil='Nuclear')   
    return rate_array

# v_0 = 233#170 #in units of km/s
# rho_dm = 0.3#0.2 #in units of Gev/cm^3 
# M_dm = 65#60 #in units of GeV
# A = 73
# sigma_0 = 7.2e-42/pb #((sigma_p * (ratio)**2 * A**2)/pb).value
# v_max = 600#in units of km/s
# period = 364 
# rate = ERF_Lewis_Freese("Ge", "CDM", "SHM", "Nucleon", 233, 0.3, 65, 7.2e-42/pb, 600, 364, 10*u.keV, 30*u.keV)
