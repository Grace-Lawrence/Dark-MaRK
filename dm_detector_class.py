#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:49:06 2019

@author: gracelawrence
"""
# coding: utf-8
from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from astropy import coordinates
import datetime
import numpy as np
from mermaid.freq_eq import reduced_mass_ratio
import mendeleev


__all__ = ["Detector", "DM", "VDF", "Nibbler","day_to_date", "calc_real_vE", 
           "calc_real_uE", "int_func"]

class Detector(object):
    def __init__(self, name,atomic_mass, M_T, atomic_number):
        """Class for variables of Detector properties"""
        self.name = name #Element of chosen detector
        self.atomic_mass = atomic_mass*u.u #Atomic mass of the target in amu
        self.atomic_number = atomic_number #Number of protons
        self.M_T = self.atomic_mass.to(GeVc2) #Reduced mass of target nucleus

class DM(object):
    def __init__(self, density, mass,cross_section, Er_range):
        """Class for variables of DM model"""
        self.density = density * (GeVc2)*(u.cm**(-3)) #Averge density of dark matter
        self.mass = mass * GeVc2 #Mass of dark matter candidate in giga-electronvolts
        self.cross_section = cross_section  #Zero momentum transfer cross section
        self.Er_range = np.logspace(-2,3,100)*u.keV #Range of recoil energies (log)
        
class VDF(object):
    def __init__(self, v_max, calc_day,v_E,u_E_x, u_E_y, u_E_z, v_0):
        """Velocity Distribution Function Properties"""
        self.v_max = v_max * u.km / u.s #Escape velocity of the halo in km/s
        self.calc_day = calc_day #Time frame for calculation in units of days (usually 365)
        self.v_E = calc_real_vE(calc_day)*u.km/u.s #The speed of the Earth through the galaxy over the given period
        self.u_E_x = calc_real_uE(calc_day,'x')*u.km/u.s #x component of the Earths velocity vector
        self.u_E_y = calc_real_uE(calc_day,'y')*u.km/u.s #y component of the Earths velocity vector
        self.u_E_z = calc_real_uE(calc_day,'z')*u.km/u.s #z component of the Earths velocity vector
        self.v_0 = v_0*(u.km/u.s) #Median velocity of dark matter velocity distribution
        
        
class Nibbler(object):
    def __init__(self, det_var, dm_var, sigma, vdf_var, v0, rho_dm, M_dm, sigma_0, v_max, period, K_form=True):
        # Save input argument
        self._det_var = det_var
        self._dm_var = dm_var
        self._sigma = sigma
        self._vdf_var = vdf_var
        self._v0 = v0
        self._rho_dm = rho_dm
        self._M_dm = M_dm
        self._sigma_0 = sigma_0
        self._v_max = v_max
        self._period = period
        self._K_form = K_form


        # Create base class objects
        self.create_objects()

    def create_objects(self):
        """
        Accepts pre-set values for DM, Detector and VDF classes
        """

        # Initialize Detector object
        detector_ob = getattr(mendeleev, self._det_var)
        if self._det_var == 'SABRE':
            det = Detector('SABRE', 150.,'M_T', int(150))
        else:
            det = Detector(detector_ob.name, detector_ob.mass, 'M_T', detector_ob.mass_number)
        self._detector = det

        # Initialize DM object
        if self._dm_var == 'CDM' and self._sigma == 'Nucleus':
            dm = DM(self._rho_dm, self._M_dm,self._sigma_0.to(u.pbarn),'Er_range')
        elif self._dm_var == 'CDM' and self._sigma == 'Nucleon':
            dm_mass = self._M_dm
            proton_mass = const.m_p#.to(GeVc2)
            target_mass = det.atomic_mass#((det.atomic_mass.value)*0.932)*GeVc2
            dm = DM(self._rho_dm, dm_mass,((self._sigma_0)*(int_func(det.atomic_number))*
                   ((reduced_mass_ratio(self._M_dm*GeVc2,target_mass,proton_mass))**2)).to(u.pbarn), 'Er_range')
        self._dm = dm
        
        # Initialize VDF object
        if self._vdf_var == 'SHM':
            vdf = VDF(self._v_max,np.linspace(0,self._period,self._period,dtype=int),'vE','u_e_x','u_e_y','u_e_z',self._v0)
        self._vdf = vdf
        
        
def day_to_date(n_day, year = 2019):
    """Turns an nth day into a date where day 0 is 01/01/2019"""
    dt = datetime.datetime(year,1,1) #Sets day 0 as January 1st 2019
    dtdelta = datetime.timedelta(days=n_day) #Finds the difference between two instances
    nday_date = dt + dtdelta #Combines the number of days with day zero
    return nday_date

def calc_real_vE(calc_day, v_E = 'Astropy'):
    """Calculate the speed of the Earth on a given day"""
    if v_E == 'Astropy':
        earth_speed = np.empty([len(calc_day)])
        for day in range(len(calc_day)):
            nday_date = day_to_date(day)
            t = Time(str(nday_date.year)+'-'+str(nday_date.month)+'-'+str(nday_date.day))
            pos_earth,vel_earth_bary = coordinates.get_body_barycentric_posvel('earth', t, ephemeris=None)
            earth_bary = coordinates.BarycentricTrueEcliptic(pos_earth.x,pos_earth.y,pos_earth.z,
                                                             vel_earth_bary.x,vel_earth_bary.y,
                                                             vel_earth_bary.z,
                                                             representation_type = 'cartesian', 
                                                             differential_type = 'cartesian')
            vel_earth = earth_bary.transform_to(coordinates.Galactocentric())
            earth_x = vel_earth.v_x.to(u.km/u.s).value
            earth_y = vel_earth.v_y.to(u.km/u.s).value
            earth_z = vel_earth.v_z.to(u.km/u.s).value
            earth_speed[day] = np.sqrt((earth_x)**2.+(earth_y)**2.+(earth_z)**2.)
    
    elif v_E == 'L&S': #Method as described in Lewin and Smith (1986)
        y = (calc_day-62)/len(calc_day) #time since March 2
        earth_speed = (244 + 15*np.sin(2*np.pi*y))*u.km/u.s
        
    elif v_E == 'L&F': 
        v_es = 29.8*u.km/u.s
        v_sh = 233*u.km/u.s
        t = calc_day/len(calc_day)
        phi_h = 2.61
        earth_speed = (np.sqrt(v_es**2+v_sh**2+(2*v_es*v_sh*np.cos(2*np.pi*t-phi_h)))).to(u.km/u.s)
        
    elif v_E == 'CERN':
        t = calc_day
        t_p = 184 #June 2nd 2020
        earth_speed = 220 * (1.05+0.07*np.cos((2*np.pi*(t-t_p))/(len(calc_day))))
    return earth_speed*u.km/u.s

def calc_real_uE(calc_day,comp):
    """Calculate the components of the Earth's velocity through the Galaxy on a given date"""
    earth_speed = np.empty([len(calc_day)])
    for day in range(len(calc_day)):
        nday_date = day_to_date(day)
        t = Time(str(nday_date.year)+'-'+str(nday_date.month)+'-'+str(nday_date.day))
        pos_earth,vel_earth_bary = coordinates.get_body_barycentric_posvel('earth', t, ephemeris=None)
        earth_bary = coordinates.BarycentricTrueEcliptic(pos_earth.x,pos_earth.y,pos_earth.z,
                                                         vel_earth_bary.x,vel_earth_bary.y,
                                                         vel_earth_bary.z,
                                                         representation_type = 'cartesian', 
                                                         differential_type = 'cartesian')
        vel_earth = earth_bary.transform_to(coordinates.Galactocentric())
        if comp == 'x':
            earth_speed[day] = vel_earth.v_x.to(u.km/u.s).value
        if comp == 'y':
            earth_y = vel_earth.v_y.to(u.km/u.s).value
            earth_speed[day] = earth_y
        if comp == 'z':
            earth_z = vel_earth.v_z.to(u.km/u.s).value
            earth_speed[day] = earth_z
    return earth_speed

def int_func(A):
    """Defines the interaction term of the target and dark matter particle"""
    return(A**2) #ASsume spin-independence for this test with an A^2 interaction term.
