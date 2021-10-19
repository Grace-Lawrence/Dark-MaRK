#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:52:42 2020

@author: gracelawrence
"""

from astropy import units as u
import astropy.constants as const
from astropy.coordinates import galactocentric_frame_defaults
galactocentric_frame_defaults.set('pre-v4.0')
import numpy as np
import scipy.stats as stats
maxwell = stats.maxwell
norm = stats.norm



from darkmark.numeric_calc_rate import spectral_func
from darkmark.freq_eq import r_kin, vmin_func,Form_Factor
from darkmark import tru, dru


__all__ = ["bootstrap_rate"]



def __num_int__(nib, ind, v_array):
    """
    Returns
    -------
    1/N times the summation of the galactocentric velocity array.

    """
    sum_vel = np.sum(v_array)
    return((1/len(v_array))*sum_vel)

def __num_vterm__(nib, v_array):
    """
     Calculates the average velocity term <v> following mathematics in the cell below.

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    v_array : TYPE
        Array of sample velocities in the galctocentric reference frame

    Returns
    -------
    <v>

    """
    return(__num_int__(nib, 1, v_array))#/__num_int__(nib, 2, v_array))



def bootstrap_rate(nib, samp_num, results_path, boostrap, n_bootstraps=0, verbose = True):
    """
    A function which integrates over the velocity distributions to generate a 
    differential rate spectrum.

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    samp_num : Integer
        The Solar Circle sample number who's VDF will be integrated over.
    results_path : String
        The folder where the samples will be outputted to.
    boostrap : Boolean
        Option of whether to implement a bootstrap resampling technique in order
        to calculate confidence intervals of the spectral function.
    n_bootstraps : Integer, optional
        The number of re-samples to perform. The default is 0.
    verbose : Boolean, optional
        Whether to print the details of the class. The default is True.

    Returns
    -------


    """
    
    if verbose:
        print(f'SAMPNUM: {samp_num}')    
        print(f'Detector \n Name: {nib._detector.name},\n Atomic Mass: {nib._detector.atomic_mass},\n Reduced Mass: {nib._detector.M_T }')
        print(f'Dark Matter Candidate \n Density: {nib._dm.density},\n Mass: {nib._dm.mass},\n Cross Section: {nib._dm.cross_section},\n Energy Range: {nib._dm.Er_range.min().value}-{nib._dm.Er_range.max()} ')
        print(f'Velocity Distribution \n Esc Velocity: {nib._vdf.v_max},\n Time Period: {nib._vdf.calc_day.max()} days,\n Median DM Velocity: {nib._vdf.v_0}')
    
    #Define minimum velocity, form factor, and vdf
    vmin = vmin_func(nib, r_kin(nib), nib._dm.Er_range)
    F_qH, q= Form_Factor(nib, LF=False, CERN=True)
    n_0 = nib._dm.density/nib._dm.mass
    for samp in range(1, samp_num+1):
        print(f'Sample {samp}')
        #Load the saved velocity distributions
        geo_vel = np.load(str(results_path)+'Velocity_Results/Sample_'+str(samp)+'/geocentric_vel.npy')
        galacto_vel = np.load(str(results_path)+'Velocity_Results/Sample_'+str(samp)+'/galactocentric_vel.npy')
        
        #Integrate the velocities to 
        spectral_func_list = np.empty([n_bootstraps, nib._vdf.calc_day.max(), len(nib._dm.Er_range)]) 
        if boostrap:
            for samp in range(0, n_bootstraps):
                rand_ind_x = (np.random.choice(np.linspace(0,len(geo_vel[0,0,:])-1,len(geo_vel[0,0,:])), size = len(geo_vel[0,0,:]), replace = True)).astype(int)
                rand_ind_y = rand_ind_x 
                rand_ind_z = rand_ind_y 
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
                year_rate = np.empty([nib._vdf.calc_day.max(), len(nib._dm.Er_range.value)])   
                for day in range(0, nib._vdf.calc_day.max()):
                    daily_rate = spectral_func(nib,F_qH,day, nib._vdf.v_E[day],vmin,speed_dist[day],norm_const, cf_ls = True)
                    year_rate[day] = daily_rate *((1./u.d)*(1./u.kg)*(1./u.keV)).to(dru)
                spectral_func_list[samp] = year_rate
        else:
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
            
            
            year_rate = np.empty([nib._vdf.calc_day.max(), len(nib._dm.Er_range.value)])   
            for day in range(0, nib._vdf.calc_day.max()):
                daily_rate = spectral_func(nib,F_qH,day, nib._vdf.v_E[day],vmin,speed_dist[day],norm_const, cf_ls = True)
                year_rate[day] = daily_rate *((1./u.d)*(1./u.kg)*(1./u.keV)).to(dru)
            spectral_func_list = year_rate
        np.save(str(results_path)+'Sample_'+str(samp)+'/spectral_function.npy', spectral_func_list )
    return 1
