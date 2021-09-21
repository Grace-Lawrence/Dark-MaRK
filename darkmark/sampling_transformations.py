# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:34:45 2019

@author: gracelawrence
"""

import astropy.units as u
import os
import numpy as np
import pynbody as pn
import astropy.coordinates as coord
from astropy.time import Time
import matplotlib.pyplot as plt
from darkmark.framewrap import simtohel
from darkmark.dm_detector_class import day_to_date

from astropy.coordinates import galactocentric_frame_defaults
galactocentric_frame_defaults.set('pre-v4.0')


__all__ = ["def_samp_coord", "calc_real_uE_pE", "sample_dm_solcirc", "vdf_load_plot"]

def def_samp_coord(target, n_samples):
    """
    Define the Sampling Coordinates in Galactocentric Coordinates

    Parameters
    ----------
    target : Data
        Simulation halo
    n_samples : Integer
        How many samples, evenly spaced, around the solar circle the user wants.

    Returns
    -------
    sample_coords : Array of floats
        Array specifying the x,y,z co-ordinates in the center of specified samples.
    theta_ang : Array of floats
        The corresponding angle around the solar circle for each of the samples.

    """
    # Define the co-ordinates of the sun
    sx, sy, sz, r = [0, 0, 0.027, 8.3] * u.kpc
    theta_ang = np.linspace(0., 2.*np.pi, n_samples) * u.radian
    theta_ang = theta_ang[0:-1]
    # Chop the last array element as it is the 2pi value which is the start
    x_comp = sx + r * np.cos(theta_ang)
    y_comp = sy + r * np.sin(theta_ang)
    z_comp = np.ones(len(theta_ang)) * sz
    sample_coords = np.dstack([x_comp.ravel(), y_comp.ravel(),
                               z_comp.ravel()])[0]
    sample_coords = np.array(sample_coords)*u.kpc
    return sample_coords, theta_ang

def calc_real_uE_pE(calc_day,comp):
    """
    Calculate the components of the Earth's velocity through the Galaxy over
    a given time period, calc_day. User can ask for specific v_x, v_y, v_z velocity
    components

    Parameters
    ----------
    calc_day : Float
        The day of the year that the user wants the position and velocity of the
        Earth on.
    comp : String
        Options are 'x','y','z', for which component of the velocity the user wants.

    Returns
    -------
    earth_pos : Vector
        The position of the Earth in Galactocentric co-ordinates.
    earth_speed : Vector
        The velocity of the Earth in Galactocentric co-ordinates.

    """
    earth_speed = np.empty([len(calc_day)])
    earth_pos = np.empty([len(calc_day)])
    for day in range(len(calc_day)):
        nday_date = day_to_date(day)
        t = Time(str(nday_date.year)+'-'+str(nday_date.month)+'-'+
                  str(nday_date.day))  
        pos_earth,vel_earth = coord.get_body_barycentric_posvel('earth',
                                                                t)
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
    """
    Calculate the Earth's peculiar motion through the galaxy

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.

    Returns
    -------
    earth_galacto : Astropy.coord Object
        Contains the Earth's peculiar motion through the galaxy'

    """
    p_E_x, u_E_x = calc_real_uE_pE(nib._vdf.calc_day,'x') 
    p_E_y, u_E_y = calc_real_uE_pE(nib._vdf.calc_day,'y')
    p_E_z, u_E_z = calc_real_uE_pE(nib._vdf.calc_day,'z')
    
    earth_bary = coord.BarycentricTrueEcliptic(x = p_E_x*u.kpc, y = p_E_y*u.kpc, 
                                               z = p_E_z*u.kpc, v_x = u_E_x*u.km/u.s, 
                                               v_y = u_E_y*u.km/u.s, v_z = u_E_z*u.km/u.s, 
                                               representation_type = 'cartesian', 
                                               differential_type = 'cartesian')
    earth_galacto = earth_bary.transform_to(coord.Galactocentric())
    return earth_galacto


def sample_dm_solcirc(nib,sample_coords, target, theta_ang, galaxy,results_path, find_boost):
    """
    Sample the dark matter components around the solar circle. Samples are radius 1kpc. 

    Parameters
    ----------
    nib : Class
        Function class containing both astrophysical and detector objects.
    sample_coords : Array of floats
        Array specifying the x,y,z co-ordinates in the center of specified samples.
    target : Data
        Simulation halo.
    theta_ang : Array of floats
        The corresponding angle around the solar circle for each of the samples.
    galaxy : String
        Options to specify the galaxy as being analogous to the Milky Way, which will
        mean that the function will either boost or restrict velocities to match 
        oberved velocity values for the Milky Way.
    results_path : String
        The folder where the samples will be outputted to.
    find_boost : Boolean
        If True, the boost to Milky Way values will be completed.

    Returns
    -------
    speed_galacto : Array of Floats
        Array of speeds of the dark matter particles in each sample, in the 
        Galactocentric reference frame.
    geo_array : Array of Floats
        Array of speeds of the dark matter particles in each sample, in the 
        Geocentric reference frame.

    """
    s = target
    earth_galacto = Earth_peculiar_motion(nib)
    earth_speed = np.sqrt((earth_galacto.v_x.value)**2+(earth_galacto.v_y.value)**2+(earth_galacto.v_z.value)**2)
    for j in range(0, len(sample_coords)):
        if not os.path.exists(results_path+'/Sample_'+str(j+1)):
            os.makedirs(results_path+'/Sample_'+str(j+1))
        solar_sphere=s[pn.filt.Sphere(radius= 1.,
                                      cen=tuple(sample_coords[j, :].value))] #take spherical samples at the specified coordinates around the solar circle.
        print(f'SAMPLE {j+1}')
        solar_sphere.rotate_z(-(theta_ang[j].to(u.degree).value+180)) #Rotate the sample back to the Earth's co-ordinate.
        
        tmp_pos_dm = solar_sphere.dm['pos'].in_units('kpc').view(type=np.ndarray) #take the positional component of the dark matter
        tmp_vel_dm = solar_sphere.dm['vel'].in_units('km s**-1').view(type=np.ndarray) #take the velocity component of the dark matter

        # if galaxy == 'milkyway':
        #     #Boost m12f simulation to the Milky Way's v_circ
        #     tmp_vel_dm[:,0] -= 1.27 #9  
        #     tmp_vel_dm[:,1] -= 23.29 #11 
        #     tmp_vel_dm[:,2] += 2.31 #9  
        
        #Transform velocities from Galactocentric -> Geo
        speed_galacto, geo_array, speed_geo_avg = simtohel(nib,tmp_pos_dm,tmp_vel_dm,'milkyway', results_path, j+1, earth_galacto)
            
        if find_boost == 'True':
            if j == 0:
                total_particle_sample_x = geo_array[0,:,:]
                total_particle_sample_y = geo_array[1,:,:]
                total_particle_sample_z = geo_array[2,:,:]
            else: 
                total_particle_sample_x = np.hstack((total_particle_sample_x,geo_array[0,:]))
                total_particle_sample_y = np.hstack((total_particle_sample_y,geo_array[1,:]))
                total_particle_sample_z = np.hstack((total_particle_sample_z,geo_array[2,:]))

    #test_simtohel() 
    if find_boost == 'True' :
        total_particle_sample_x_avg = np.sum(total_particle_sample_x, axis=1)/len(total_particle_sample_x[0,:])#np.average(total_particle_sample_x, axis=1)
        total_particle_sample_y_avg = np.sum(total_particle_sample_y, axis=1)/len(total_particle_sample_x[0,:])#np.average(total_particle_sample_y,axis=1)
        total_particle_sample_z_avg = np.sum(total_particle_sample_z, axis=1)/len(total_particle_sample_x[0,:])#np.average(total_particle_sample_z,axis=1)
        X Boost = np.mean((total_particle_sample_x_avg-earth_galacto.v_x.value))
        Y Boost = np.mean(total_particle_sample_y_avg-earth_galacto.v_y.value)
        Z Boost = np.mean(total_particle_sample_z_avg-earth_galacto.v_z.value)
        


    
    return speed_galacto, geo_array

def vdf_load_plot(maximum_day,results_path, samp_num):
    plt.clf()
    fig, axs = plt.subplots(4, 2, figsize=(20, 30), sharex=True, sharey=True,tight_layout=True)
    n_bins = 100
    axes = np.array(([0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]))
    for sample in range(0, samp_num-1): 
        i = axes[sample][0]
        j = axes[sample][1]
        print(i,j)
        galacto_vel_samp1 = np.load(str(results_path)+'/Sample_'+str(sample+1)+'/galactocentric_vel.npy')
        galacto_vel_1samp = np.sqrt(galacto_vel_samp1[0]**2.+galacto_vel_samp1[1]**2.+galacto_vel_samp1[2]**2.)

        # helio_vel_samp1 = np.load(str(results_path)+'/Sample_'+str(sample+1)+'/barycentric_vel.npy')
        # helio_vel_1samp = np.sqrt(helio_vel_samp1[0]**2.+helio_vel_samp1[1]**2.+helio_vel_samp1[2]**2.)
    
        geo_vel_samp1 = np.load(str(results_path)+'/Sample_'+str(sample+1)+'/geocentric_vel.npy')
        geo_vel_1samp = geo_vel_samp1[:,196,:]
        geo_vel_1samp = np.sqrt(geo_vel_1samp[0]**2.+geo_vel_1samp[1]**2.+geo_vel_1samp[2]**2.)
        
        axs[i,j].hist(geo_vel_1samp, bins=n_bins,color='grey',density=True, label = 'Geo')
        # axs[i,j].hist(helio_vel_1samp, bins=n_bins,histtype=u'step',linewidth=2,color='red',density=True, label = 'Bary')
        axs[i,j].hist(galacto_vel_1samp, bins=n_bins,histtype=u'step',linewidth=2,color='black',density=True, label = 'Galacto')

        axs[i,j].set_title("Sample"+str(sample+1))

    axs[0,1].legend(fontsize=25)
    plt.show()
    return 1
