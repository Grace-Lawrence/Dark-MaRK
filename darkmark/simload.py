# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:30:28 2019
@author: gracelawrence
"""

"""Module For Loading, Smoothing and Centering Simulations"""
try:
    import commah
    import_commah = True
except ImportError:
    import_commah = False
    
import numpy as np
import astropy.units as u
import pynbody as pn
import matplotlib.pyplot as plt

__all__ = ["import_sim", "apply_x_inversion", "apply_centre", "apply_smth", 
           "visualize_dm", "visualize_gas", "set_cosmology", "load_sim_profile"]

def import_sim(file_path):
    """
    Loads the Simulation File and provides the redshift, mass, loadable keys
    and families of the file.

    Parameters
    ----------
    file_path : String
        The file path to the location of the simulation files.

    Returns
    -------
    target : Simulation
        The simulation instance to be used for analysis.
    s : Simulation
        Another simulation instance with original mass data type.

    """
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
    """
    Inversion of the spatial and velocity components to flip rotation of spiral
    galaxy.

    Parameters
    ----------
    target : Data
        Simulation instance.

    Returns
    -------
    None.

    """
    target['pos'][:,0] *= -1
    target['vel'][:,0] *= -1 
    
def apply_centre(target, **v_cen): 
    """
    Uses angular momentum and velocity centres to align the galaxy such that
    the user views if 'face-on'.

    Parameters
    ----------
    target : Data
        Simulation instance.
    **v_cen : Array of floats
        Velocity centre values to help centre the galaxy.

    Returns
    -------
    None.

    """
    # Set FaceOn and Center
    pn.analysis.angmom.faceon(target, cen_size='1 kpc', disk_size='50 kpc', 
                              move_all=True, vcen=(v_cen[0],v_cen[1], v_cen[2]))


def apply_smth(latte, s, target):
    """
    Apply Smoothing Lengths and Centre

    Parameters
    ----------
    latte : String
        Description of the type of simulation. Latte is specific to the Latte suite
        of FIRE simulations (Andrew R. Wetzel et al 2016 ApJL 827 L23). Other simulations
        can be manually included. Otherwise, softening lengthscale eps is assigned 
        and applied from simulation properties
    s : Simulation
        Another simulation instance with original mass data type.
    target : Data
        Simulation instance.
    Returns
    -------
    target : Data
        Smoothed simulation instance with Softening lengthscale eps applied.
    s : Data
        Smoothed simulation instance with Softening lengthscale eps applied.

    """
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


def visualize_dm(target):
    """
    Create Face-On Image of the Dark Matter Component

    Parameters
    ----------
    target : Data
        Smoothed simulation instance with Softening lengthscale eps applied.

    Returns
    -------
    None.

    """
    pn.plot.image(target.dm, width='60 kpc', cmap=plt.cm.Greys,
                      resolution=2500, units='Msol kpc^-2', qtytitle=r'$\Sigma$')
    plt.show()
    plt.clf()


def visualize_gas(target):
    """
    Create Face-On Image of the Gas Components

    Parameters
    ----------
    target : Data
        Smoothed simulation instance with Softening lengthscale eps applied.

    Returns
    -------
    None.

    """
    pn.plot.image(target.g, width='60 kpc', cmap=plt.cm.Oranges,
                      resolution=2500, units='Msol kpc^-2', qtytitle=r'$\Sigma$')
    plt.show()
    plt.clf()


def set_cosmology(s, cosmology):
    """
    Define Cosmology Values, Critial Density, Virial Radius and
    Scale Radius. List of available cosmologies available at 
    https://github.com/astroduff/commah/blob/master/commah/cosmology_list.py

    Parameters
    ----------
    s : Simulation
        Another simulation instance with original mass data type.
    cosmology : String
        Specificy which survery the user would like to assume for the cosmology. 
        Default is WMAP1 .

    Raises
    ------
    ImportError
        Checks is the cosmology package Commah .

    Returns
    -------
    concentration : Float
        Concentration value of the galaxy.
    rho_crit : Float
        Critical density of the galaxy.

    """
    if not import_commah:
        raise ImportError("commah package unavailable. Please install with 'pip install darkmark[extras]'")
    output = commah.run('cosmology', zi=0., Mi=8.34e12, z=[0.0])
    concentration = output['c']
    rho_crit = cosmology.rho_crit(s, z=0., unit=(pn.units.Msol/(pn.units.h*pn.units.Mpc**3)))
    rho_crit = rho_crit*(u.solMass)/(u.Mpc)**3
    print("The mass of dm halo is; ","{:e}".format(s.gas['mass'].sum().in_units('Msol')),"$\ M_{\odot}$")
    print('Critical Density: ', rho_crit)
    return concentration, rho_crit


def load_sim_profile(latte, file_path, visualize, invert_rot, cosmology = 'WMAP1'):
    """
    Load the simulation and apply centering and smooething to it

    Parameters
    ----------
    latte : String
        Description of the type of simulation. Latte is specific to the Latte suite
        of FIRE simulations (Andrew R. Wetzel et al 2016 ApJL 827 L23). Other simulations
        can be manually included. Otherwise, softening lengthscale eps is assigned 
        and applied from simulation properties.
    file_path : String
        The file path to the location of the simulation files.
    visualize : Boolean
        Whether or not to generate graphical visualizations of the Gas and DM
        components of the galaxy. 
    invert_rot : Boolean
        Whether or not to invert the galaxt along the x-axis to make it 'rotate'
        in the opposite direction.
    cosmology : String, optional
        The default is 'WMAP1'.

    Returns
    -------
    concentration : Float
        Concentration value of the galaxy.
    rho_crit : Float
        Critical density of the galaxy.
    target : Data
        The simulation instance to be used for analysis.
    s : Data
        Another simulation instance with original mass data type.

    """
    target, s = import_sim(latte, file_path)
    target, s = apply_smth(latte, s, target)
    apply_centre(target)
    if invert_rot == True: #reverse the direction of rotation of the galaxy
        apply_x_inversion(target)
    concentration, rho_crit = set_cosmology(s, cosmology)
    if visualize:
        visualize_dm(target)
        visualize_gas(target)
    return concentration, rho_crit, target, s

