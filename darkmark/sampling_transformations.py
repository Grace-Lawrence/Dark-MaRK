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


__all__ = ["def_samp_coord", "calc_real_uE_pE", "sample_dm_solcirc"]

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
    n_samples = n_samples+1
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
        Contains the Earth's peculiar motion through the galaxy

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
        if not os.path.exists(str(results_path)+'Velocity_Results/Sample_'+str(j+1)):
            os.makedirs(str(results_path)+'Velocity_Results/Sample_'+str(j+1))
        
        solar_sphere=s[pn.filt.Sphere(radius= 1.,
                                      cen=tuple(sample_coords[j, :].value))] #take spherical samples at the specified coordinates around the solar circle.
        solar_sphere.rotate_z(-(theta_ang[j].to(u.degree).value+180)) #Rotate the sample back to the Earth's co-ordinate.
        
        tmp_pos_dm = solar_sphere.dm['pos'].in_units('kpc').view(type=np.ndarray) #take the positional component of the dark matter
        tmp_vel_dm = solar_sphere.dm['vel'].in_units('km s**-1').view(type=np.ndarray) #take the velocity component of the dark matter
        
        #Transform velocities from Galactocentric -> Geo
        speed_galacto, geo_array, speed_geo_avg = simtohel(nib,tmp_pos_dm,tmp_vel_dm,'milkyway', results_path, j+1, earth_galacto)
    return speed_galacto, geo_array

