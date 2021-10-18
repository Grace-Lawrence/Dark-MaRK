#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:30:28 2019

@author: gracelawrence
"""
import numpy as np
import os
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import galactocentric_frame_defaults
galactocentric_frame_defaults.set('pre-v4.0')

__all__ = ['simtohel']


def simtohel(nib,pos, vel, galaxy, results_path, samp_num,earth_galacto): 
    """
    Bespoke array to take halo centred sim positions and convert to ICRS 
    pos and vel. Values assumed to be kpc, km/s

    Parameters
    ----------
    nib : Class
    Function class containing both astrophysical and detector objects.
    pos : Array of floats
        Array of x,y,z positions associated with the sampled particles, in the galactocentric reference frame.
    vel : Array of floats
        Array of x,y,z velocities associated with the sampled particles, in the galactocentric reference frame.
    ggalaxy : String
        Options to specify the galaxy as being analogous to the Milky Way, which will
        mean that the function will either boost or restrict velocities to match 
        oberved velocity values for the Milky Way.  
    results_path : String
        The folder where the samples will be outputted to.
    samp_num : Integer
        Number of samples around the solar circle. 
    earth_galacto : Astropy.coord Object
        Contains the Earth's peculiar motion through the galaxy

    Returns
    -------
    speed_galacto : Array of floats
        Array of speed values of sample particles in the galactocentric frame.
    speed_geo : Array of floats
       Array of speed values of sample particles in the geocentric frame, evaluated for 
       each day of the specified time period.
    average_speed_geo :  Array of floats
        The mean of the geocentric speed distributions.

    """
    #Set Galactocentric Frame
    xyz = coord.Galactocentric(x = pos[:,0]*u.kpc, y = pos[:,1]*u.kpc, 
                                z = pos[:,2]*u.kpc, v_x = vel[:,0]*u.km/u.s,
                                v_y = vel[:,1]*u.km/u.s, 
                                v_z = vel[:,2]*u.km/u.s, 
                                representation_type = CartesianRepresentation)
    xyz.representation_type = 'cartesian'
    #Transform to Geocentric
    geo_x = []
    geo_y = []
    geo_z = []
    for day in range(0,len(nib._vdf.calc_day)):
        geo_x.append(xyz.v_x.value + earth_galacto[day].v_x.value) 
        geo_y.append(xyz.v_y.value + earth_galacto[day].v_y.value)
        geo_z.append(xyz.v_z.value + earth_galacto[day].v_z.value)
    speed_geo = np.array([np.array(geo_x),np.array(geo_y),np.array(geo_z)])

    #Find the Mean
    mean_geo_x = []
    mean_geo_y = []
    mean_geo_z = []
    for day in range(0,len(nib._vdf.calc_day)):
        mean_geo_x.append(np.mean(geo_x[day]))
        mean_geo_y.append(np.mean(geo_y[day]))
        mean_geo_z.append(np.mean(geo_z[day]))
    mean_geo_x = np.array(mean_geo_x)
    mean_geo_y = np.array(mean_geo_y)
    mean_geo_z = np.array(mean_geo_z)

    #Calculate Speeds 
    average_speed_geo = np.sqrt((mean_geo_x)**2+(mean_geo_y)**2+(mean_geo_z)**2)

    speed_galacto = np.sqrt((xyz.v_x.value)**2+(xyz.v_y.value)**2
                            +(xyz.v_z.value)**2)
    save_velocity_info(xyz, speed_geo, results_path, samp_num)
    
    return speed_galacto, speed_geo, average_speed_geo


def save_velocity_info(galacto, geo, results_path, samp_num):
    """
    Save the velocity outputs information

    Parameters
    ----------
    galacto : Array of floats
        Array of speed values of sample particles in the galactocentric frame.
    geo : Array of floats
        Array of speed values of sample particles in the geocentric frame, evaluated for 
       each day of the specified time period.
    results_path : String
        The folder where the samples will be outputted to.
    samp_num : Integer
        Number of samples around the solar circle. 

    """
    if not os.path.exists(str(results_path)+'Velocity_Results/Sample_'+str(samp_num)):
        os.makedirs(str(results_path)+'Velocity_Results/Sample_'+str(samp_num))
    #Save the Galactocentric distribution
    galacto_array = np.array([galacto.v_x,galacto.v_y,galacto.v_z])
    np.save(os.path.join(str(results_path)+'Velocity_Results/Sample_'+str(samp_num), 'galactocentric_vel'),
            galacto_array)

    #Save the Geocentric distribution
    np.save(os.path.join(str(results_path)+'Velocity_Results/Sample_'+str(samp_num), 'geocentric_vel'),
            geo)
    return 1  
