#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:52:42 2020

@author: gracelawrence
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.constants as const
from mermaid.dm_detector_class import Nibbler
from mermaid.numeric_calc_rate import spectral_func, Summation_Int_Model
from mermaid.velocity_int import annual_mod, find_amplitude
from mermaid.freq_eq import r_kin, vmin_func,Form_Factor, R_0, E_0

n_samples = 9
latte = True
pointmass = 35182.46875
nsidehealpy = 4
sim_path = "./Sims/snapshot_600.1.hdf5"
vdf_path = "/fred/oz071/glawrence/DarkMark/results_cache/"
results_path = "../results_cache_laptop/New/"
if not os.path.exists('../results_cache_laptop/New/'):
    os.makedirs(results_path)
n_bootstraps = 1000
energy_range = np.linspace(0,100,100)

__all__ = ["bootstrap_rate", "conf_int, bootstrap_rate_geo"]

def __vterm__(nib):
    """Calculates the average velocity term <v> following mathematics in the cell below."""
    v_term = (__vterm_numerator__(nib._vdf.v_max.value,nib))/(__vterm_denominator__(nib._vdf.v_max.value,nib))
    return v_term*u.km/u.s

def __num_int__(nib, ind, v_array):
    sum_vel = np.sum(v_array)
    return((1/len(v_array))*sum_vel)

def __num_vterm__(nib, v_array):
    return(__num_int__(nib, 1, v_array))#/__num_int__(nib, 2, v_array))

def conf_int(data_input, conf_perc):
    data_input = np.sort(data_input) #sort the y values from low to high
    array_len = len(data_input) #how many index values are there?
    sigma = 0.6826
    sample_ind = np.round(np.array([ array_len*(1. - sigma)/2., array_len*0.5,  array_len*(1. + sigma)/2. ]))
    low_CI, med, high_CI = data_input[ sample_ind.astype(int) ]
    # end, start = st.t.interval(alpha=conf_perc, df=len(data_input)-1, loc=np.median(data_input), scale=st.sem(data_input)) 
    return low_CI, med, high_CI

def bootstrap_rate(samp_num, detvar, dmvar, sigma_cs, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, min_ee, max_ee, recoil, Freese):
    dmvar=("CDM")
    vdf_var =("SHM")
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max,period)    
    print(f'Detector \n Name: {nib._detector.name},\n Atomic Mass: {nib._detector.atomic_mass},\n Reduced Mass: {nib._detector.M_T }')
    print(f'Dark Matter Candidate \n Density: {nib._dm.density},\n Mass: {nib._dm.mass},\n Cross Section: {nib._dm.cross_section},\n Energy Range: {nib._dm.Er_range.min().value}-{nib._dm.Er_range.max()} ')
    print(f'Velocity Distribution \n Esc Velocity: {nib._vdf.v_max},\n Time Period: {nib._vdf.calc_day.max()} days,\n Median DM Velocity: {nib._vdf.v_0}')
    #Define minimum velocity, form factor, and vdf
    vmin = vmin_func(nib, r_kin(nib), nib._dm.Er_range)
    F_qH, q= Form_Factor_Alan(nib, LF=False, CERN=True)
    # F_qH, q, qrn, Er_n = helm_form(nib)
    galacto_vel = np.load(str(results_path)+'Sample_'+str(samp_num)+'/galactocentric_vel.npy')
    samples_x = np.empty([n_bootstraps, len(galacto_vel[0,:])])
    samples_y = np.empty([n_bootstraps,len(galacto_vel[1,:])])
    samples_z = np.empty([n_bootstraps,len(galacto_vel[2,:])])
    amplitdue_list = np.empty([n_bootstraps, len(energy_range)])
    for samp in range(0, n_bootstraps):
        if samp == 0:
            samples_x[samp] = galacto_vel[0,:]
            samples_y[samp] = galacto_vel[1,:]
            samples_z[samp] = galacto_vel[2,:]
            plt.clf()
            plt.hist(samples_x[samp])
            plt.hist(samples_y[samp])
            plt.hist(samples_z[samp])
            plt.title('BS')
            plt.show()
        else:  
            samples_x[samp] = np.random.choice(galacto_vel[0,:], size = len(galacto_vel[0,:]), replace = True)
            samples_y[samp] = np.random.choice(galacto_vel[1,:], size = len(galacto_vel[1,:]), replace = True)
            samples_z[samp] = np.random.choice(galacto_vel[2,:], size = len(galacto_vel[2,:]), replace = True)
        dru_rate = Summation_Int_Model(nib,samples_x[samp],samples_y[samp],samples_z[samp],F_qH,vmin,min_ee, max_ee, samp_num, results_path, model = True, cf_ls=True, recoil = recoil, Freese=Freese)        
        amp = []
        for energy in energy_range:
            am_Ge = annual_mod(nib, dru_rate, min_ee = energy*u.keV, max_ee =energy*u.keV, recoil='Nuclear')
            amplitude = find_amplitude(nib, am_Ge, fit = True)
            amp.append(amplitude)
        amplitdue_list[samp] = np.array(amp)
    amp_median = np.median(amplitdue_list, axis = 0)
    plt.clf()
    plt.plot(energy_range, amplitdue_list[0], label = 'Original')
    plt.plot(energy_range, amp_median, label = 'bootstrap')
    plt.legend()
    plt.show()
    return amplitdue_list

#@profile
def bootstrap_rate_geo(samp_num, detvar, dmvar, sigma_cs, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, min_ee, max_ee, recoil, Freese):
    if samp_num == 'total':
        for i in range(0,8):
            if i == 0:
                geo_vel = np.load(str(vdf_path)+'Sample_'+str(i+1)+'/geocentric_vel_365.npy')
                print(f'Sample 1 Geo: {geo_vel.shape}')
                galacto_vel = np.load(str(vdf_path)+'Sample_'+str(i+1)+'/galactocentric_vel_365.npy')
                print(f'Sample 1 Galacto: {galacto_vel.shape}')
            else:
                geo_vel_add = np.load(str(vdf_path)+'Sample_'+str(i+1)+'/geocentric_vel_365.npy')
                galacto_vel_add = np.load(str(vdf_path)+'Sample_'+str(i+1)+'/galactocentric_vel_365.npy')
                print(f'Sample {i}: {geo_vel.shape}')
                geo_vel = np.concatenate((geo_vel,geo_vel_add),axis = 2)
                galacto_vel = np.concatenate((galacto_vel,galacto_vel_add),axis = 1)
                print(f'Concatenated Geo: {geo_vel.shape}')
                print(f'Concatenated Galacto: {galacto_vel.shape}')
    np.save('/fred/oz071/glawrence/DarkMark/results_cache/Sample_tot/total_geo_dist.npy', geo_vel)
    np.save('/fred/oz071/glawrence/DarkMark/results_cache/Sample_tot/total_galacto_dist.npy', galacto_vel)
    dmvar=("CDM")
    vdf_var =("SHM")
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max,period)
#    print(f'SAMPNUM: {samp_num}')    
#    print(f'Detector \n Name: {nib._detector.name},\n Atomic Mass: {nib._detector.atomic_mass},\n Reduced Mass: {nib._detector.M_T }')
 #   print(f'Dark Matter Candidate \n Density: {nib._dm.density},\n Mass: {nib._dm.mass},\n Cross Section: {nib._dm.cross_section},\n Energy Range: {nib._dm.Er_range.min().value}-{nib._dm.Er_range.max()} ')
  #  print(f'Velocity Distribution \n Esc Velocity: {nib._vdf.v_max},\n Time Period: {nib._vdf.calc_day.max()} days,\n Median DM Velocity: {nib._vdf.v_0}')
    #Define minimum velocity, form factor, and vdf
    vmin = vmin_func(nib, r_kin(nib), nib._dm.Er_range)
    F_qH, q= Form_Factor(nib, LF=False, CERN=True)
    ## F_qH, q, qrn, Er_n = helm_form(nib)
    #geo_vel = np.load(str(vdf_path)+'Sample_'+str(samp_num)+'/geocentric_vel_365.npy')
    #galacto_vel = np.load(str(vdf_path)+'Sample_'+str(samp_num)+'/galactocentric_vel_365.npy')
    #year_rate = np.empty([period, len(nib._dm.Er_range.value)])
    amplitdue_list = np.empty([n_bootstraps, len(energy_range)])
    spectral_func_list = np.empty([n_bootstraps, period, len(energy_range)]) 
    n_0 = nib._dm.density/nib._dm.mass
 
    for samp in range(0, n_bootstraps):
        print(f'Core: {samp_num}, BootStrap Num: {samp}')
        rand_ind_x = (np.random.choice(np.linspace(0,len(geo_vel[0,0,:])-1,len(geo_vel[0,0,:])), size = len(geo_vel[0,0,:]), replace = True)).astype(int)
        rand_ind_y = rand_ind_x #(np.random.choice(np.linspace(0,len(geo_vel[1,0,:])-1,len(geo_vel[1,0,:])), size = len(geo_vel[1,0,:]), replace = True)).astype(int)
        rand_ind_z = rand_ind_y #(np.random.choice(np.linspace(0,len(geo_vel[2,0,:])-1,len(geo_vel[2,0,:])), size = len(geo_vel[2,0,:]), replace = True)).astype(int)
        if samp == 0:
            u_x = galacto_vel[0,:]
            u_y = galacto_vel[1,:]
            u_z = galacto_vel[2,:]
            speed_dist_galacto = np.sqrt((u_x)**2+(u_y)**2+(u_z)**2)
         
            v_term =  __num_vterm__(nib, speed_dist_galacto)*(u.km/u.s)
            R0 = (((const.N_A*(nib._dm.cross_section)*n_0*v_term)/(nib._detector.atomic_number*(u.g/u.mol)))).to(tru)
            E0 = (0.5*nib._dm.mass*v_term**2.*(np.pi/4)).to(u.keV)
            norm_const =((R0.to(tru)*v_term)/(E0.to(u.keV)*r_kin(nib)))*(np.pi/4)#coefficient
           
            samples_x = geo_vel[0,:,:]
            samples_y = geo_vel[1,:,:]
            samples_z = geo_vel[2,:,:]
            speed_dist = np.sqrt(samples_x**2+samples_y**2+samples_z**2)
        else:  
            u_x = galacto_vel[0,rand_ind_x]
            u_y = galacto_vel[1,rand_ind_y]
            u_z = galacto_vel[2,rand_ind_z]
            speed_dist_galacto = np.sqrt((u_x)**2+(u_y)**2+(u_z)**2)
        
            v_term =  __num_vterm__(nib, speed_dist_galacto)*(u.km/u.s)
            R0 = (((const.N_A*(nib._dm.cross_section)*n_0*v_term)/(nib._detector.atomic_number*(u.g/u.mol)))).to(tru)
            E0 = (0.5*nib._dm.mass*v_term**2.*(np.pi/4)).to(u.keV)
            norm_const =((R0.to(tru)*v_term)/(E0.to(u.keV)*r_kin(nib)))*(np.pi/4)#coefficient
        
            samples_x = geo_vel[0,:,rand_ind_x]
            samples_y = geo_vel[1,:,rand_ind_y]
            samples_z = geo_vel[2,:,rand_ind_z]
            speed_dist = (np.sqrt(samples_x**2+samples_y**2+samples_z**2)).T
        year_rate = np.empty([period, len(nib._dm.Er_range.value)])   
        for day in range(0, period):
            daily_rate = spectral_func(nib,F_qH,day, nib._vdf.v_E[day],vmin,speed_dist[day],norm_const, cf_ls = True)
            year_rate[day] = daily_rate *((1./u.d)*(1./u.kg)*(1./u.keV)).to(dru)
        spectral_func_list[samp] = year_rate
        amp = []
        for energy in energy_range:
            am_Ge = annual_mod(nib, year_rate*dru, min_ee = energy*u.keV, max_ee =energy*u.keV, recoil='Nuclear')
            amplitude = find_amplitude(nib, am_Ge, fit = True)
            amp.append(amplitude)
        amplitdue_list[samp] = np.array(amp)
    amp_median = np.median(amplitdue_list, axis = 0)
    
    return amplitdue_list, spectral_func_list

