# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:45:19 2019

@author: gracelawrence
"""
from __future__ import print_function, absolute_import, division
import pynbody as pn
import numpy as np
import astropy.units as u
import pynbody.analysis.profile as profile
import matplotlib.pyplot as plt

from mermaid import show_plot


"""Module For Generating Cosmology and Profiles of the Snapshots"""


__all__ = ["dens_profiles", "vcirc_profiles", "calc_NFW",
           "dens_comp_profiles", "profiles"]


def dens_profiles(target, show_plot):
    """Observe the density functions of the different elements of the galaxy"""

    minr, maxr = 0.5, 150.
    h = 5.
    ps = profile.Profile(target.s, min=minr, max=maxr, type='lin', ndim=3)
    pdm = profile.Profile(target.dm, min=minr, max=maxr, type='lin', ndim=3)
    soldisc = target[pn.filt.Disc(maxr, h)]
    disc_ps = profile.Profile(soldisc.s, min=minr, max=maxr, type='lin',
                              ndim=3)
    disc_pdm = profile.Profile(soldisc.dm, min=minr, max=maxr, type='lin',
                               ndim=3)
    stars_r_i = disc_ps['rbins']
    stars_r_j = np.roll(disc_ps['rbins'], 1)
    stars_r_j[0] = minr
    dm_r_j = np.roll(disc_pdm['rbins'], 1)
    dm_r_j[0] = minr

    stars_disc_volume = np.pi*(stars_r_i - stars_r_j)**2*h
    dm_disc_volume = np.pi * (stars_r_i - stars_r_j) ** 2 * h

    if show_plot is True:
        plt.plot(ps['rbins'], ps['mass'], label='stars')
        plt.plot(pdm['rbins'], pdm['mass'], label='dm')
        plt.ylabel('Mass')
        plt.xlabel('Radius')
        plt.legend()

    return pdm, disc_ps, disc_pdm, stars_disc_volume, dm_disc_volume, ps


def vcirc_profiles(target, show_plot):
    """Plot the Circular Velocity Functions for Each Simulation Component"""
    if show_plot is True:
        pd = pn.analysis.profile.Profile(target.d, min=.0, max=100, nbins = 90, type='lin')
        pg = pn.analysis.profile.Profile(target.g, min=.0, max=100, nbins = 90, type='lin')
        p = pn.analysis.profile.Profile(target, min=.0, max=100, nbins = 90, type='lin')
        ps = pn.analysis.profile.Profile(target.s, min=.0, max=100, nbins = 90, type='lin')

        plt.clf()
        plt.plot(pd['rbins'], pd['v_circ'], label='dark')
        plt.plot(pg['rbins'], pg['v_circ'], label='gas')
        plt.plot(p['rbins'], p['v_circ'], label='total')
        plt.plot(ps['rbins'], ps['v_circ'], label='stars')
        plt.axvline(x=8.3, color='black', label='8.3kpc')
        plt.xlabel('$R$ [kpc]')
        plt.ylabel('$v_{circ}$ [km/s]')
        plt.legend()
        plt.show()


def calc_NFW(concentration, r_S, rho_crit, pdm, v, s):
    """Calculate the NFW Profile for the Halo"""
    r = pdm['rbins'] * u.kpc
    g_c = 1./((np.log(1+concentration)-(concentration/(1+concentration))))
    char_dens = (v*concentration**3.*g_c)/3.
    nfw_prof = (rho_crit*char_dens) / (r/r_S).value*(1.+(r/r_S).value**2)

    print('Concentration', concentration)
    print('g(c)', g_c)
    print('Characteristic Density', char_dens[0])
    print('Scale Radius', r_S.value)
    print('Density', rho_crit)
    return nfw_prof


def dens_comp_profiles(pdm, disc_ps, disc_pdm, stars_disc_volume,
                       dm_disc_volume, ps, lineall, lineg, lines, linedm,
                       nfw_prof):
    """Plot the Density Profiles and the NFW Profile for the Dark, Stellar
    and Gas Components"""

    stars_disc_ps_dens = ((disc_ps['mass'].in_units('Msol'))/stars_disc_volume)
    stars_disc_pdm_dens = ((disc_pdm['mass'].in_units('Msol'))/dm_disc_volume)
    minr = 0.5
    x = np.arange(8., 9., 0.1)
    y1 = 0
    y2 = 10e12
    if show_plot is True:
        for plotn in np.arange(6):
            f, ax = plt.subplots()
            ax.loglog()
            ax.set_xlabel('R [kpc]')
            ax.set_ylabel(r'$\rho$ [M$_{\odot}$ kpc$^{-3}$]')
            ax.set_xlim(minr, 130)
            plt.fill_between(x, y1, y2, color='grey', alpha='0.5')
            if plotn > 0:
                yrange = ax.get_ylim()
                ax.plot([8.3, 8.3], yrange, '-', color='red', alpha=0.5)
                ax.plot([8., 8.], yrange, '-', color='grey', alpha=0.5)
                ax.plot([9., 9.], yrange, '-', color='grey', alpha=0.5)
            if plotn > 1:
                ax.plot(ps['rbins'].in_units('kpc'), ps['density'],
                        linestyle=lines, color='k', label='Stars', linewidth=2)
                ax.plot([8.3, 8.3], yrange, '-', color='red', alpha=0.5)
            if plotn > 2:
                ax.plot(pdm['rbins'].in_units('kpc'), pdm['density'],
                        linestyle=linedm, color='k', label='DM', linewidth=2)
            if plotn > 3:
                ax.plot(disc_ps['rbins'], stars_disc_ps_dens, linestyle=lines,
                        color='darkorange', label='Stars disc', linewidth=2)
            if plotn > 4:
                ax.plot(disc_pdm['rbins'], stars_disc_pdm_dens,
                        linestyle=linedm, color='indigo', label='DM disc',
                        linewidth=2)
                ax.plot(pdm['rbins'], nfw_prof[0, :], color='k', label='NFW',
                        linewidth=10)
                ax.plot([8.3, 8.3], yrange, '-', color='red', alpha=0.5)
            yrange = ax.get_ylim()
            # ax.plot([8.3,8.3],yrange,'-',color='red', alpha=0.5)
            plt.legend()
            # ax.set_xlim(2,100)
            # plt.ylim(0,10e12)
            plt.savefig('total_rho'+str(plotn)+'.pdf', bbox_inches='tight')

            plt.show()
            
def get_radial_velocity_profile(target):
    ps = pn.analysis.profile.Profile(target.d, min=6, max=10, nbins = 100, type='lin')
    test_tangential = ps['vcxy']#['vt']
    ps = pn.analysis.profile.Profile(target.s, min=6, max=10, nbins = 100, type='lin')
    test_radial = ps['vrxy']
    radius = ps['rbins']
    plt.clf()
    plt.plot(radius,test_radial, label ='rad')
    plt.plot(radius,test_tangential, label ='tan')
    plt.title('Radial Velocity')
    plt.legend()
    plt.show()
    
    plt.clf()
    plt.hist(test_radial, label ='rad')
    plt.hist(test_tangential, label ='tan')
    plt.title('Radial Velocity')
    plt.legend()
    plt.show()
    return 1


def profiles(target, s, concentration, r_S, rho_crit, v, show_plot):
    # pdm, disc_ps, disc_pdm, stars_disc_volume, dm_disc_volume, ps = dens_profiles(target, show_plot)
    # vcirc_profiles(target, show_plot)
    get_radial_velocity_profile(target)
    # nfw_prof = calc_NFW(concentration, r_S, rho_crit, pdm, v, s)
    #dens_comp_profiles(pdm, disc_ps, disc_pdm, stars_disc_volume, dm_disc_volume, ps,lineall, lineg, lines, linedm, nfw_prof)