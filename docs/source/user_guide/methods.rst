Methods
===============
The expected signal in a direct detection experiment relies on two classes of input parameters; Astrophysical and Particle. It's important to correctly model both in order to create realistic expectactions for experiments given the galactic dark matter environment, the dark matter particle model, and the type of detector being used. 

The following equation (Green2012_) is the standard form of the differential rate equation, which expresses the expected detection rate of dark matter in units of counts per day, per kilogram of detector, per kev (energy);

.. math::
    \frac{dR}{dE_R} = \frac{\sigma_p \rho_0}{2 \mu_p^2 m_{\chi}} A^2 F^2(E) \int_{v_{min}}^{\infty} \frac{f(v, v_E)}{v} d^3v

Where :math:`\sigma_p` is the interaction cross-section of the dark matter particle, :math:`\rho_0` is the local dark matter density, :math:`\mu_p` is the WIMP-proton reduced mass, :math:`m_{\chi}` is the WIMP mass, :math:`A^2` is the interaction factor, :math:`F^2(E)`, is the form factor, and :math:`f(v, v_E)` is the geocentric velocity distribution.

  
Halo Model and Cosmology
--------------------------
The cosmology and halo model being used have a large impact on the detection rate as they inform the speed distribution of dark matter. The dark matter cosmology used helps to define the large scale structure, or 'clumpiness' of dark matter while the hierachichal assembly history of halos may give rise to dark matter substructures like stellar streams, or debris flows. These features can cause increases in the local dark matter density, in the speed distribution of dark matter, and subsequently strongly impact on the detection rate. Dark-MaRK assumes :math:`\Lambda CDM`.  


Velocity Distribution
-----------------------
The velocity distribution of dark matter :math:`f(v,v_E)`, is integrated over a velocity range in order to define the spectral rate function, and is one of the most important input parameters in this particular dark matter calculation. Modelling its form accurately is vitally important, and highlights the need for realistic, high resolution, hydrodynamic simulations of Milky Way analogues. 


Frame Transformation
-----------------------
The velocity distribution needs to undergo a transformation from the galactocentric reference frame, :math:`f_{gal}`, to the geocentric reference frame, :math:`f_{geo}` (the velocity distribution as observed from Earth). For this, the galactocentric reference frame undergoes a galilean boost; 

.. math::
    f_{geo}(\vec{v}, t) = {f}_{gal}(\vec{v}+\vec{v}_{Earth}(t))

Where :math:`v_E` is the velocity of the Earth. Dark-MaRK defines :math:`v_E` via Astropy_'s Astronomical Coordinate Systems sub-package. 

Event Rate
--------------
The event rate is then performed as an integral over the velocity distribution, to generate the spectral rate function, :math:`\frac{dR}{dE_R}`. Assuming the traditional Standard Halo Model, the velocity distribution of dark matter in the solar neighbourhood takes the form of an analytical Maxwell Boltzmann. This can be integrated analytically using; 

.. math::
    \frac{dR}{dE_R} = \frac{2\, N_0 \, n_0 \, \sigma}{A \, k \, M_D \, r} \int_{v_{min}}^{v_{max}} \frac{1}{v} \, f_{geo}(\vec{v},\vec{v}_E) \, d^3\vec{v}.


When implementing numerical simulations, or arrays of velocities instead of analytical forms, the integral can be transformed into a model independent summation over the velocities. This summation form is implemented in Dark-MaRK.

.. _Green2012: https://arxiv.org/abs/1112.0524
.. _Astropy: https://www.astropy.org/

