#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:52:42 2020

@author: gracelawrence
"""
import numpy as np
import os
import matplotlib.pyplot as plt
# from mermaid.numeric_calc_rate import Numeric_Calc_Rate, spectral_func, Summation_Int_Model
from astropy import units as u
import astropy.constants as const
from astropy.time import Time
import astropy.coordinates as coord
from mermaid.dm_detector_class import Nibbler, day_to_date
from mermaid.numeric_calc_rate import spectral_func, Summation_Int_Model
from mermaid.velocity_int import annual_mod, find_amplitude
import scipy.stats as stats
maxwell = stats.maxwell
norm = stats.norm
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import galactocentric_frame_defaults
galactocentric_frame_defaults.set('pre-v4.0')



vdf_path = "/fred/oz071/glawrence/DarkMark/results_cache/"
results_path = "../results_cache_laptop/New/"
if not os.path.exists('../results_cache_laptop/New/'):
    os.makedirs(results_path)


n_bootstraps = 10000
energy_range = np.linspace(1,100,100)

__all__ = ["bootstrap_rate", "conf_int", "bootstrap_rate_geo", "bootstrap_rate_geo_MB", "bootstrap_rate_geo_MB_solarcircle"]

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



def bootstrap_rate_geo(samp_num, detvar, dmvar, sigma_cs, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, min_ee, max_ee, recoil, Freese):
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
    geo_vel = np.load(str(vdf_path)+'Sample_'+str(samp_num)+'/geocentric_vel_365.npy')
    galacto_vel = np.load(str(vdf_path)+'Sample_'+str(samp_num)+'/galactocentric_vel_365.npy')
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
            am_Ge = annual_mod(nib, year_rate*dru, min_ee = energy*u.keV, max_ee =energy*u.keV, recoil=recoil)
            amplitude = find_amplitude(nib, am_Ge, fit = True)
            amp.append(amplitude)
        amplitdue_list[samp] = np.array(amp)
    amp_median = np.median(amplitdue_list, axis = 0)
    
    return amplitdue_list, spectral_func_list


def calc_real_uE_pE(calc_day,comp):
    """Calculate the components of the Earth's position +velocity through the 
    Galaxy on a given date"""
    earth_speed = np.empty([len(calc_day)])
    earth_pos = np.empty([len(calc_day)])
    for day in range(len(calc_day)):
        nday_date = day_to_date(day)
        t = Time(str(nday_date.year)+'-'+str(nday_date.month)+'-'+
                  str(nday_date.day))  
        obstime = (Time('J2010') + calc_day*u.day)
        pos_earth,vel_earth = coord.get_body_barycentric_posvel('earth',t)
        if comp == 'x':
            earth_speed[day] = vel_earth.x.to(u.km/u.s).value
            earth_pos[day] = pos_earth.x.to(u.kpc).value    
        if comp == 'y':
            earth_y = vel_earth.y.to(u.km/u.s).value
            earth_speed[day] = earth_y
            earth_pos[day] = pos_earth.y.to(u.kpc).value
        if comp == 'z':
            earth_z = vel_earth.z.to(u.km/u.s).value
            earth_speed[day] = earth_z
            earth_pos[day] = pos_earth.z.to(u.kpc).value
    return earth_pos, earth_speed


def Earth_peculiar_motion(nib):
    v_lsr = np.array([0,220,0])#v_sun = np.array([-10,5,7])
    v_sun = np.array([11.1,12.2,7.3])
    v_bary = v_lsr + v_sun
    
    p_E_x, u_E_x = calc_real_uE_pE(nib._vdf.calc_day,'x') 
    p_E_y, u_E_y = calc_real_uE_pE(nib._vdf.calc_day,'y')
    p_E_z, u_E_z = calc_real_uE_pE(nib._vdf.calc_day,'z')
    
    earth_bary = coord.BarycentricTrueEcliptic(x = p_E_x*u.kpc, y = p_E_y*u.kpc, 
                                               z = p_E_z*u.kpc, v_x = u_E_x*u.km/u.s, 
                                               v_y = u_E_y*u.km/u.s, v_z = u_E_z*u.km/u.s, 
                                               representation_type = 'cartesian', 
                                               differential_type = 'cartesian')
    earth_galacto = earth_bary.transform_to(coord.Galactocentric())


    v_E_pm_x = earth_galacto.v_x.value - v_bary[0]
    v_E_pm_y = earth_galacto.v_y.value - v_bary[1]
    v_E_pm_z = earth_galacto.v_z.value - v_bary[2]

    return earth_galacto

def galacto_to_geo(nib,vel,earth_galacto): 
    """ Bespoke array to take halo centred sim positions and convert to ICRS 
    pos and vel. Values assumed to be kpc, km/s"""
    method = 'first'
    #Set Galactocentric Frame
    xyz = coord.Galactocentric(x = np.ones([len(vel[0,:])])*u.kpc, y = np.ones([len(vel[1,:])])*u.kpc, 
                                z = np.ones([len(vel[2,:])])*u.kpc, v_x = vel[0,:]*u.km/u.s,
                                v_y = vel[1,:]*u.km/u.s, 
                                v_z = vel[2,:]*u.km/u.s, 
                                representation_type = CartesianRepresentation)
    xyz.representation_type = 'cartesian'
    print(f'Galacto x: {np.mean(xyz.v_x)}, Galacto y: {np.mean(xyz.v_y)}, Galacto z: {np.mean(xyz.v_z)}')
    #Transform to Geocentric
    geo_x = []
    geo_y = []
    geo_z = []
    for day in range(0,len(nib._vdf.calc_day)):
        geo_x.append(xyz.v_x.value + earth_galacto[day].v_x.value) #Needs to be + in order for it to be a boost. Intuitively, we need to have the - 
        #print the earth galacto x,y,z, to check 
        geo_y.append(xyz.v_y.value + earth_galacto[day].v_y.value)
        geo_z.append(xyz.v_z.value + earth_galacto[day].v_z.value)
    speed_geo = np.array([np.array(geo_x),np.array(geo_y),np.array(geo_z)])
    
    return  speed_geo


def bootstrap_rate_geo_MB(samp_num, detvar, dmvar, sigma_cs, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, min_ee, max_ee, recoil, Freese):
    dmvar=("CDM")
    vdf_var =("SHM")
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max,period)
    #Define minimum velocity, form factor, and vdf
    vmin = vmin_func(nib, r_kin(nib), nib._dm.Er_range)
    F_qH, q= Form_Factor(nib, LF=False, CERN=True)
    earth_galacto = Earth_peculiar_motion(nib)
    amplitdue_list = np.empty([n_bootstraps, len(energy_range)])
    spectral_func_list = np.empty([n_bootstraps, period, len(energy_range)]) 
    n_0 = nib._dm.density/nib._dm.mass
        
    #Define Velocity Distribution
    galacto_vel = np.load(str(vdf_path)+'Sample_'+str(samp_num)+'/galactocentric_vel_365.npy')
    galacto_vel_speed = np.sqrt(galacto_vel[0]**2.+galacto_vel[1]**2.+galacto_vel[2]**2.)
    params_x = norm.fit(galacto_vel[0], floc=0, scale=np.median(galacto_vel[0]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0
    print(f'Sample: {samp_num}, Params X: {params_x}') 
    params_y = norm.fit(galacto_vel[1], floc=0, scale=np.median(galacto_vel[1]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0 
    print(f'Params Y: {params_y}') 
    params_z = norm.fit(galacto_vel[2], floc=0, scale=np.median(galacto_vel[2]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0  
    print(f'Params Z: {params_z}') 
    exit()
    galacto_vel_MB_x = np.random.normal(loc=0.0, scale=params_x[1], size=len(galacto_vel[0]))
    galacto_vel_MB_y = np.random.normal(loc=0.0, scale=params_y[1], size=len(galacto_vel[1]))
    galacto_vel_MB_z = np.random.normal(loc=0.0, scale=params_z[1], size=len(galacto_vel[2]))
    galacto_vel_MB = np.vstack([galacto_vel_MB_x, galacto_vel_MB_y,galacto_vel_MB_z])
    np.save('/fred/oz071/glawrence/DarkMark/MB_results_cache/Sample'+str(samp_num)+'_Galacto_new.npy',galacto_vel_MB)
    print(f'Core test: {galacto_vel_MB.shape}')
    #print(f'galacto vel MB size: {galacto_vel_MB.shape}')
    geo_vel = galacto_to_geo(nib, galacto_vel_MB, earth_galacto)
    np.save('/fred/oz071/glawrence/DarkMark/MB_results_cache/Sample'+str(samp_num)+'_Geo_new.npy',geo_vel)
    amplitdue_list = np.empty([n_bootstraps, len(energy_range)])
    spectral_func_list = np.empty([n_bootstraps, period, len(energy_range)]) 
    n_0 = nib._dm.density/nib._dm.mass
    for samp in range(0, n_bootstraps):
        print(f'Core: {samp_num}, BootStrap Num: {samp}')
        rand_ind_x = (np.random.choice(np.linspace(0,len(geo_vel[0,0,:])-1,len(geo_vel[0,0,:])), size = len(geo_vel[0,0,:]), replace = True)).astype(int)
        rand_ind_y = rand_ind_x #(np.random.choice(np.linspace(0,len(geo_vel[1,0,:])-1,len(geo_vel[1,0,:])), size = len(geo_vel[1,0,:]), replace = True)).astype(int)
        rand_ind_z = rand_ind_y #(np.random.choice(np.linspace(0,len(geo_vel[2,0,:])-1,len(geo_vel[2,0,:])), size = len(geo_vel[2,0,:]), replace = True)).astype(int)
        if samp == 0:
            u_x = galacto_vel_MB[0,:]
            u_y = galacto_vel_MB[1,:]
            u_z = galacto_vel_MB[2,:]
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
            u_x = galacto_vel_MB[0,:][rand_ind_x]
            u_y = galacto_vel_MB[1,:][rand_ind_y]
            u_z = galacto_vel_MB[2,:][rand_ind_z]
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
            am_Ge = annual_mod(nib, year_rate*dru, min_ee = energy*u.keV, max_ee =energy*u.keV, recoil=recoil)
            amplitude = find_amplitude(nib, am_Ge, fit = True)
            amp.append(amplitude)
        amplitdue_list[samp] = np.array(amp)
    amp_median = np.median(amplitdue_list, axis = 0)
    
    return amplitdue_list, spectral_func_list



def bootstrap_rate_geo_MB_solarcircle(samp_num, detvar, dmvar, sigma_cs, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, min_ee, max_ee, recoil, Freese):
    dmvar=("CDM")
    vdf_var =("SHM")
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max,period)
    #Define minimum velocity, form factor, and vdf
    vmin = vmin_func(nib, r_kin(nib), nib._dm.Er_range)
    F_qH, q= Form_Factor(nib, LF=False, CERN=True)
    earth_galacto = Earth_peculiar_motion(nib)
    amplitdue_list = np.empty([n_bootstraps, len(energy_range)])
    spectral_func_list = np.empty([n_bootstraps, period, len(energy_range)]) 
    n_0 = nib._dm.density/nib._dm.mass
        
    #Define Velocity Distribution
    galacto_vel = np.load('/fred/oz071/glawrence/DarkMark/results_cache/Sample_tot/total_galacto_dist.npy')
    galacto_vel_speed = np.sqrt(galacto_vel[0]**2.+galacto_vel[1]**2.+galacto_vel[2]**2.)
    params_x = norm.fit(galacto_vel[0], floc=0, scale=np.median(galacto_vel[0]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0
    print(f'Sample: {samp_num}, Params X: {params_x}') 
    params_y = norm.fit(galacto_vel[1], floc=0, scale=np.median(galacto_vel[1]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0 
    print(f'Params Y: {params_y}') 
    params_z = norm.fit(galacto_vel[2], floc=0, scale=np.median(galacto_vel[2]))   #fit the galactocentric distribution to a maxwell boltzmann to determine v0  
    print(f'Params Z: {params_z}') 
    galacto_vel_MB_x = np.random.normal(loc=0.0, scale=params_x[1], size=len(galacto_vel[0]))
    galacto_vel_MB_y = np.random.normal(loc=0.0, scale=params_y[1], size=len(galacto_vel[1]))
    galacto_vel_MB_z = np.random.normal(loc=0.0, scale=params_z[1], size=len(galacto_vel[2]))
    galacto_vel_MB = np.vstack([galacto_vel_MB_x, galacto_vel_MB_y,galacto_vel_MB_z])
    np.save('/fred/oz071/glawrence/DarkMark/MB_results_cache/Sample'+str(samp_num)+'_Galacto_new.npy',galacto_vel_MB)
    geo_vel = galacto_to_geo(nib, galacto_vel_MB, earth_galacto)
    np.save('/fred/oz071/glawrence/DarkMark/MB_results_cache/Sample'+str(samp_num)+'_Geo_new.npy',geo_vel)
    spectral_func_list = np.empty([n_bootstraps, period, len(energy_range)]) 
    n_0 = nib._dm.density/nib._dm.mass
    for samp in range(0, n_bootstraps):
        print(f'Core: {samp_num}, BootStrap Num: {samp}')
        rand_ind_x = (np.random.choice(np.linspace(0,len(geo_vel[0,0,:])-1,len(geo_vel[0,0,:])), size = len(geo_vel[0,0,:]), replace = True)).astype(int)
        rand_ind_y = rand_ind_x #(np.random.choice(np.linspace(0,len(geo_vel[1,0,:])-1,len(geo_vel[1,0,:])), size = len(geo_vel[1,0,:]), replace = True)).astype(int)
        rand_ind_z = rand_ind_y #(np.random.choice(np.linspace(0,len(geo_vel[2,0,:])-1,len(geo_vel[2,0,:])), size = len(geo_vel[2,0,:]), replace = True)).astype(int)
        if samp == 0:
            u_x = galacto_vel_MB[0,:]
            u_y = galacto_vel_MB[1,:]
            u_z = galacto_vel_MB[2,:]
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
            u_x = galacto_vel_MB[0,:][rand_ind_x]
            u_y = galacto_vel_MB[1,:][rand_ind_y]
            u_z = galacto_vel_MB[2,:][rand_ind_z]
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
    
    return spectral_func_list
