import astropy.units as u 
import astropy.constants as const  
import numpy as np 
import numexpr as ne
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,mark_inset)
import os
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.interpolate import InterpolatedUnivariateSpline 
maxwell = stats.maxwell


from darkmark.freq_eq import E_0, r_kin
from darkmark.velocity_int import annual_mod, find_amplitude, find_DAMA_fit
from darkmark.annual_mod_fits import fit_func_S, fit_curve  


dru = 1./(u.d*u.kg*u.keV)
tru = 1./(u.d*u.kg)



def vdf_load_plot(maximum_day,results_path, sample, save = True):
    plt.clf() 
    n_bins = 100 
    geo_col = ["dimgray", "indianred", "navajowhite", "lightgreen", "lightskyblue", "thistle", "pink", "palegoldenrod"] 
    gal_col = ["black", "darkred", "darkorange", "forestgreen", "royalblue", "mediumorchid", "deeppink", "goldenrod"] 
    galacto_vel = np.load(str(results_path)+'/Velocity_Results/Sample_'+str(sample+1)+'/galactocentric_vel.npy') 
    galacto_vel_speed = np.sqrt(galacto_vel[0]**2.+galacto_vel[1]**2.+galacto_vel[2]**2.)
    geo_vel = np.load(str(results_path)+'/Velocity_Results/Sample_'+str(sample+1)+'/geocentric_vel.npy') 
    geo_vel_speed_tot = np.sqrt(geo_vel[0,:,:]**2+(geo_vel[1,:,:])**2+(geo_vel[2,:,:])**2) 
    avg = np.average(geo_vel_speed_tot, axis=1) 
    maximum_day = np.where(avg == avg.max())[0][0] 
    geo_vel_pd = geo_vel[:,maximum_day,:] 
    geo_vel_speed = np.sqrt(geo_vel_pd[0]**2.+geo_vel_pd[1]**2.+geo_vel_pd[2]**2.) 
    params = maxwell.fit(galacto_vel_speed, floc=0, scale=np.median(galacto_vel_speed)) 
    x = np.linspace(0, galacto_vel_speed.max(), len(galacto_vel_speed)) 
    plt.plot(x, maxwell.pdf(x, *params), '--', lw=2, label = 'Maxwellian Fit', color = 'black') 
    print(f'Maximum Day: {maximum_day}, geo_vel_pd: {geo_vel_pd.shape}') 
    plt.hist(geo_vel_speed, bins=n_bins,color=geo_col[sample],density=True, label = 'Geocentric', alpha = 0.6) 
    plt.hist(galacto_vel_speed, bins=n_bins,histtype=u'step',linewidth=3,color=gal_col[sample],density=True, label = 'Galactocentric') 
    plt.title('Sample '+str(sample))  
    plt.legend(loc='upper right') 
    plt.ylabel(r'$\ f(v) ~(normed) ~ (km~s^{-1})$') 
    plt.xlabel(r'$\ v~ (km~s^{-1})$') 
    if save:
        plt.savefig(str(results_path)+'Velocity_Results/VDF_comparison.png') 
    plt.show() 
    plt.clf() 
    return 1 

def __num_vterm__(nib, v_array):
    return(__num_int__(nib, 1, v_array))
    data_input_sorted = np.sort(data_input) #sort the y values from low to high
    array_len = len(data_input) #how many index values are there?
    sigma =conf_perc
    sample_ind = np.round(np.array([ array_len*(1. - sigma)/2., array_len*0.5,  array_len*(1. + sigma)/2. ]))
    sample_ind[sample_ind < 0] = 0
    sample_ind[sample_ind > len(data_input)-1] = len(data_input)-1
    low_CI, med, high_CI = data_input_sorted[sample_ind.astype(int)]
    return low_CI, med, high_CI

def __num_int__(nib, ind, v_array):
    sum_vel = np.sum(v_array)
    return((1/len(v_array))*sum_vel)

def spectral_func_plot(nib, sample, results_path):  
    spec_func =  np.load(str(results_path)+'/Velocity_Results/Sample_'+str(sample+1)+'/spectral_function.npy') 
    annual_average = np.average(spec_func[0,:], axis = 0) 
    solarcirlc_galacto = np.load(str(results_path)+'/Velocity_Results/Sample_'+str(sample+1)+'/galactocentric_vel.npy')  
    u_x = solarcirlc_galacto[0,:] 
    u_y = solarcirlc_galacto[0,:] 
    u_z = solarcirlc_galacto[0,:] 
    
    n_0 = nib._dm.density/nib._dm.mass 
    speed_dist_galacto = ne.evaluate("sqrt((u_x)**2.+(u_y)**2.+(u_z)**2.)") 
    v_term =  __num_vterm__(nib, speed_dist_galacto)*(u.km/u.s) 
    E0 = (0.5*nib._dm.mass*v_term**2.*(np.pi/4)).to(u.keV) 
    R0 = (((const.N_A*(nib._dm.cross_section)*n_0*v_term)/(nib._detector.atomic_number*(u.g/u.mol)))).to(tru) 
    r = r_kin(nib) 
    E = nib._dm.Er_range.value 
 
    dEdR1 = spec_func[0,:] 
    dEdR2 = spec_func[90,:] 
    dEdR3 = spec_func[180,:] 
    dEdR4 = spec_func[270,:] 
          
    fig, ax1 = plt.subplots(1, figsize=(30, 20)) 
    ax1.plot(E,dEdR1, label = 'Day 0 ', color = 'black', lw = 2) 
    ax1.plot(E,dEdR2, label = 'Day 90 ', color = 'red',lw = 2) 
    ax1.plot(E,dEdR3, label = 'Day 180 ', color = 'blue',lw = 2) 
    ax1.plot(E,dEdR4, label = 'Day 270 ', color = 'green',lw = 2) 
    ax1.legend(prop={'size': 15}, ncol=2)# plt.legend(bbox_to_anchor=(0.8, 0.5)) 
    ax1.set_xlabel(r'$E_R$ (keV)') 
    ax1.set_ylabel(r'$\frac{dR}{dE_R} (dru)$') 
    ax1.tick_params(axis='x')  
    ax1.tick_params(axis='y')  
    ax1.set_xlim(0,90)       
    # Create a set of inset Axes: these should fill the bounding box allocated to them. 
    ax2 = plt.axes([0.0,0.0,0.5,0.5])       
    # Manually set the position and relative size of the inset axes within ax1 
    ip = InsetPosition(ax1, [0.5,0.4 ,0.4,0.4]) 
    ax2.set_axes_locator(ip)      
    # Mark the region corresponding to the inset axes on ax1 and draw lines 
    # in grey linking the two axes. 
    ax2.plot((E),dEdR1-annual_average, label = 'Day 0 ', color = 'black', lw = 2) 
    ax2.plot((E),dEdR2-annual_average, label = 'Day 90 ', color = 'red',lw = 2) 
    ax2.plot((E),dEdR3-annual_average, label = 'Day 180 ', color = 'blue',lw =2) 
    ax2.plot((E),dEdR4-annual_average, label = 'Day 270 ', color = 'green',lw = 2) 
    ax2.tick_params(axis='x')  
    ax2.tick_params(axis='y')  
    ax2.set_xlim(0,90) 
    ax2.set_xlabel(r'$E_R$ (keV)') 
    ax2.set_ylabel(r'$\frac{dR}{dE_R}-\widetilde{\frac{dR}{dE_R}} (dru)$') 
    plt.savefig(str(results_path)+'Velocity_Results/spectral_function.png') 
    plt.show() 
    return 1

def annual_modulation_plot(nib, sample, min_ee, max_ee, recoil, results_path):
    fig, axs = plt.subplots(1)
    geo_col = ["dimgray", "indianred", "navajowhite", "lightgreen", "lightskyblue", "thistle", "pink", "palegoldenrod"]
    gal_col = ["black", "darkred", "darkorange", "forestgreen", "royalblue", "mediumorchid", "deeppink", "goldenrod"]
    
    #Generate All Annual Modulation Signals
    spec_func = np.load(str(results_path)+'/Velocity_Results/Sample_'+str(sample+1)+'/spectral_function.npy') 
    am_Ge = annual_mod(nib, spec_func*tru, min_ee = min_ee*u.keV, max_ee =max_ee*u.keV, recoil=recoil)
    
    #Fit the model
    S_0, S_m, t_0 = find_DAMA_fit(nib, am_Ge, fit = True)
    print(f'Fit parameters - S_0: {S_0}, S_m: {S_m}, t_0: {t_0}')
    axs.plot(nib._vdf.calc_day, fit_func_S(nib._vdf.calc_day, S_0, S_m, t_0),label='Model Fit', color='black')
    axs.plot(am_Ge, color = gal_col[sample], label = 'Sim Data')
    axs.tick_params(axis='x') 
    axs.tick_params(axis='y') 
    axs.legend() 
    plt.xlabel('Time (day)')
    if recoil == 'E_Equiv':
        plt.ylabel(r'$\ \frac{R_{band}}{\Delta E_{ee}}~(dru[cpd^{-1}~kg^{-1}~keV_{ee}^{-1}])$')
    elif recoil == 'Nuclear':
        plt.ylabel(r'$\ \frac{R_band}{\Delta E_R}~(dru[cpd^{-1}~kg^{-1}~keV^{-1}])$')
    plt.savefig(str(results_path)+'Velocity_Results/annual_modulation.png')
    plt.show()
    return 1
