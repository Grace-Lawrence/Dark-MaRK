#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:56:31 2020

@author: gracelawrence
"""
from __future__ import print_function, absolute_import, division
import astropy.units as u
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
from mermaid.velocity_int import thint
from mermaid.freq_eq import r_kin, helm_form, vmin_func
from mermaid.dm_detector_class import Nibbler
import numexpr as ne
from mermaid.freq_eq import Form_Factor


#Define Globals 
GeVc2 = u.def_unit('GeVc2', u.GeV/(const.c)**2)
keVc2 = u.def_unit('keVc2', u.keV/(const.c)**2)
dru = 1./(u.d*u.kg*u.keV)
tru = 1./(u.d*u.kg)
pb = 1e-36
solar_x = 232.24 #check the coordinates
solar_y = 11.1
solar_z = 7.25
#Nsteps = 100000 #Number of velocity samples

__all__ = ["Numeric_Calc_Rate", "spectral_func", "Summation_Int_Model"]


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
    # return((1/len(v_array))*np.mean(v_array)**ind*sum_vel)
    
def __num_vterm__(nib, v_array):
    return(__num_int__(nib, 1, v_array))#/__num_int__(nib, 2, v_array))

def reduced_m(M_x,M_n):
    a = (M_x*M_n/(M_x+M_n))
    return a

def freese_vE(nib):
    v_es = 29.8*u.km/u.s
    v_sh = 233*u.km/u.s
    t = nib._vdf.calc_day/365
    phi_h = 2.61
    v_eh = (v_es**2)+(v_sh**2)+(2*v_es*v_sh*np.cos(2*np.pi*t-phi_h))
    vel_earth = np.sqrt(v_eh)
    return vel_earth

def Summation_Int_Model(nib, u_x,u_y,u_z,F_qH,vmin,min_ee, max_ee, samp_num,results_path,model, cf_ls, recoil, Freese):
    """Calls the summation method for each specified day"""
    #Define Coefficient Terms
    n_0 = nib._dm.density/nib._dm.mass
    speed_dist_galacto = ne.evaluate("sqrt((u_x)**2.+(u_y)**2.+(u_z)**2.)")
    v_term =  __num_vterm__(nib, speed_dist_galacto)*(u.km/u.s)
    R0 = (((const.N_A*(nib._dm.cross_section)*n_0*v_term)/(nib._detector.atomic_number*(u.g/u.mol)))).to(tru)
    E0 = (0.5*nib._dm.mass*v_term**2.*(np.pi/4)).to(u.keV)
    if Freese: 
        mr = reduced_m(65*GeVc2,(73*0.93)*GeVc2)
        # norm_const = ((nib._dm.density*nib._dm.cross_section)/(2*mr**2*nib._dm.mass))*const.c**6
        norm_const = ((2*nib._dm.density*nib._dm.cross_section)/(4*nib._dm.mass*mr**2)).to(u.km/(u.d*u.keV*u.kg*u.s))#*const.c**6
        print(f'Freese Norm Const: {norm_const.to(u.km/(u.d*u.keV*u.kg*u.s))}')
    else:
        norm_const =((R0.to(tru)*v_term)/(E0.to(u.keV)*r_kin(nib)))*(np.pi/4)#coefficient
    rkin = r_kin(nib)
    v_E = nib._vdf.v_E 
    v_lsr = np.array([0,220,0])
    v_sun = np.array([11.1,12.2,7.3])
    v_bary = v_lsr + v_sun
    #For each day of the year calculate vE
    year_rate = np.empty([len(v_E), len(nib._dm.Er_range.value)])
    for index, vE in enumerate(v_E):
        if model == True:
            day = v_E.value.tolist().index(vE.value)
            if Freese:
                vel_earth = freese_vE(nib)
                speed = ne.evaluate("sqrt((u_x)**2.+(u_y)**2.+(u_z)**2.)")
                speed_dist = speed - vel_earth[day].value
            #Transform Using Earth Velocities
                # v_x = u_x + (solar_x+nib._vdf.u_E_x[day].value)
                # v_y = u_y + (solar_y+nib._vdf.u_E_y[day].value)
                # v_z = u_z + (solar_z+nib._vdf.u_E_z[day].value)
            else:
                v_x = u_x + (nib._vdf.u_E_x[day].value)
                v_y = u_y + (nib._vdf.u_E_y[day].value)
                v_z = u_z + (nib._vdf.u_E_z[day].value)
                speed_dist = np.sqrt(v_x**2+v_y**2+v_z**2)
        elif model == False: 
            day = v_E.value.tolist().index(vE.value)
            speeddist = np.load(str(results_path)+'/Sample_'+str(samp_num)+'/geocentric_vel.npy')
            speed_dist = np.sqrt(speeddist[0,day,:]**2+speeddist[1,day,:]**2+speeddist[2,day,:]**2)
        #Perform the Summation Method
        daily_rate = spectral_func(nib,F_qH,day, vE,vmin,speed_dist,norm_const, cf_ls)
        year_rate[index] =  daily_rate
    year_rate = (year_rate*((1./u.d)*(1./u.kg)*(1./u.keV)).to(dru))    
    
    differece_rs = abs(year_rate[180,:] - year_rate[0,:])
    cross_over = (np.where(differece_rs == differece_rs.min()))[0]
    cross_over_normed = (nib._dm.Er_range/(E0*rkin))[cross_over]
    cross_over_energy = (nib._dm.Er_range)[cross_over]
    print(f'Difference Max/Min: {differece_rs.max()}, {differece_rs.min()}')
    print(f'Cross Over En: {cross_over_energy}')
    print(f'Cross Over Normalized: {cross_over_normed}')
    print(f'E0: {E0}')
    print(f'Rkin: {rkin}')
    # plt.clf()
    # plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*year_rate[0,:],s=1)
    # plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*year_rate[90,:],s=1)
    # plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*year_rate[180,:],s=1)
    # plt.scatter((nib._dm.Er_range/(E0*rkin)), (E0*rkin)/R0*year_rate[270,:],s=1)
    # plt.xlabel(r'$ \frac{E}{E_0r}$')
    # plt.ylabel(r'$ \frac{E_0r}{R_0}\frac{dR}{dE_r}$')
    # plt.xlim(0,10)
    # plt.title(f'Seasonal Variation of Rate Spectrum {nib._detector.name}')
    # plt.show()
    
    return year_rate

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

def Numeric_Calc_Rate(detvar, sigma_cs, min_ee, max_ee, v0, period, rho_dm, M_dm, sigma_0, v_max, samp_num,results_path, recoil, model, Freese):
    #Define the classes
    dmvar=("CDM")
    vdf_var =("SHM")
    nib = Nibbler(detvar, dmvar, sigma_cs, vdf_var, v0,rho_dm, M_dm, sigma_0, v_max,period)
    print(f'Detector \n Name: {nib._detector.name},\n Atomic Mass: {nib._detector.atomic_mass},\n Reduced Mass: {nib._detector.M_T }')
    print(f'Dark Matter Candidate \n Density: {nib._dm.density},\n Mass: {nib._dm.mass},\n Cross Section: {nib._dm.cross_section},\n Energy Range: {nib._dm.Er_range.min().value}-{nib._dm.Er_range.max()} ')
    print(f'Velocity Distribution \n Esc Velocity: {nib._vdf.v_max},\n Time Period: {nib._vdf.calc_day.max()} days,\n Median DM Velocity: {nib._vdf.v_0}')
    
    #Define minimum velocity, form factor, and vdf
    vmin = (vmin_func(nib, r_kin(nib), nib._dm.Er_range))
    F_qH, q= Form_Factor(nib, LF=False, CERN=True)
    # F_qH, q, qrn, Er_n = helm_form(nib)
    
    if model == True:
        mu,sigma = 0, 270/np.sqrt(2)#nib._vdf.v_0.value/np.sqrt(2) #mean and standard deviation
        mu_z,sigma_z = 0, (nib._vdf.v_0.value-30)/np.sqrt(2) #mean and standard deviation

        x = (np.random.normal(mu, sigma, Nsteps))
        y = (np.random.normal(mu, sigma, Nsteps))
        z = (np.random.normal(mu, sigma, Nsteps))
    elif model == False:
        vdf =  np.load(str(results_path)+'/Sample_'+str(samp_num)+'/galactocentric_vel.npy')
        x = vdf[0,:]
        y = vdf[1,:]
        z = vdf[2,:]
    
    #Call integration function
    dru_rate = Summation_Int_Model(nib,x,y,z,F_qH,vmin,min_ee, max_ee, samp_num, 
                                   results_path,model, cf_ls=True, recoil = recoil, Freese=Freese)
    return dru_rate

# import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plt

# data = vdf[0,:] 
# mean,std=norm.fit(data)

# plt.hist(data, bins=30, density=True)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# y = norm.pdf(x, mean, std)
# plt.plot(x, y)
# plt.show()
