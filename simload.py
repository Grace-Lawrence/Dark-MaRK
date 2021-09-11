from __future__ import print_function, absolute_import, division
import commah
# import pynbody.analysis.cosmology as cosmology
import numpy as np
import astropy.units as u
import pynbody as pn
import matplotlib.pyplot as plt

# !/usr/bin/env python2
#  -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:30:28 2019
@author: gracelawrence
"""

"""Module For Loading, Smoothing and Centering Simulations"""

__all__ = ["import_sim", "apply_x_inversion", "apply_centre", "apply_smth", 
           "visualize_dm", "visualize_gas", "set_cosmology", "load_sim_profile"]


def import_sim(latte, file_path):
    """Load the Simulation File"""
    s = pn.load(file_path)
    s.physical_units()
    target = s
    print("This Galaxy Has a RedShift of z = ", round(s.properties['Redshift']))
    print("The mass of halo is; ","{:e}".format(s['mass'].sum().in_units('Msol')),
          "$\ M_{\odot}$")
    print('Loadable Keys ', target.loadable_keys())
    print('Families ', target.families())

    # Reassignment of mass data type
    mass = np.asanyarray(target['mass'], dtype=target['pos'].dtype)
    del target['mass']
    target['mass'] = mass

    return target, s
    
def apply_x_inversion(target):
    target['pos'][:,0] *= -1
    target['vel'][:,0] *= -1 
    
def apply_centre(target, **v_cen):  
    # Set FaceOn and Center
    pn.analysis.angmom.faceon(target, cen_size='1 kpc', disk_size='50 kpc', 
                              move_all=True, vcen=(v_cen[0],v_cen[1], v_cen[2]))


def apply_smth(latte, s, target):
    """Apply Smoothing Lengths and Centre"""
    if latte is True:
        target.g['eps'] = pn.array.SimArray(np.double(1.), 'pc h**-1')
        target.s['eps'] = pn.array.SimArray(np.double(4.), 'pc h**-1')
        target.dm['eps'] = pn.array.SimArray(np.double(40.), 'pc h**-1')
    else:
        eps = pn.array.SimArray(np.double(200.), 'pc h**-1')
        if s.properties['Redshift'] < 2.9:
            s.properties.update({'eps': eps})
        else:
            s.properties.update({'eps': eps*(1.+2.9)/(1.+np.double(s.properties
                                 ['Redshift']))})
        print("Softening lengthscale eps is ", s.properties['eps'])

        target.g['eps'] = s.properties['eps']
        target.s['eps'] = s.properties['eps']
        target.dm['eps'] = s.properties['eps']

    return target, s


def visualize_dm(target, s):
    """Create Face-On Image of the Dark Matter Component"""
    pn.plot.image(target.dm, width='60 kpc', cmap=plt.cm.Greys,
                      resolution=2500, units='Msol kpc^-2', qtytitle=r'$\Sigma$')
    plt.show()
    plt.clf()


def visualize_gas(target, s):
    """Create Face-On Image of the Gas Components"""
    pn.plot.image(target.g, width='60 kpc', cmap=plt.cm.Oranges,
                      resolution=2500, units='Msol kpc^-2', qtytitle=r'$\Sigma$')
    plt.show()
    plt.clf()


def set_cosmology(s, cosmology):
    """Define Cosmology Values, Critial Density, Virial Radius and
    Scale Radius"""
    output = commah.run('cosmology', zi=0., Mi=8.34e12, z=[0.0]) #List of available cosmologies available at https://github.com/astroduff/commah/blob/master/commah/cosmology_list.py
    concentration = output['c']
    rho_crit = cosmology.rho_crit(s, z=0., unit=(pn.units.Msol/(pn.units.h*pn.units.Mpc**3)))
    rho_crit = rho_crit*(u.solMass)/(u.Mpc)**3
    print("The mass of dm halo is; ","{:e}".format(s.gas['mass'].sum().in_units('Msol')),"$\ M_{\odot}$")
    print('Critical Density: ', rho_crit)
    return concentration, rho_crit


def load_sim_profile(latte, file_path, visualize, invert_rot, cosmology = 'WMAP1'):
    "Load the simulation and apply centering and smooething to it"
    target, s = import_sim(latte, file_path)
    target, s = apply_smth(latte, s, target)
    apply_centre(target)
    if invert_rot == True: #reverse the direction of rotation of the galaxy
        apply_x_inversion(target)
    concentration, r_S, rho_crit, v = set_cosmology(s, cosmology)
    if visualize:
        visualize_dm(target,s,file_path)
        visualize_gas(target,s,file_path)
    return concentration, r_S, rho_crit, v, target, s
