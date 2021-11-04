.. Dark-MaRK documentation master file, created by
   sphinx-quickstart on Mon Oct 18 14:28:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dark-Mark's documentation!
=====================================
Dark Mark (Dark Matter Rate Kalculator) is an open source Python 3 package created to generate bespoke predictions for direct detection dark matter experiments. Dark Mark allows users to generate these predictions using input galaxy simulations or arrays of velocities from a given model. The package allows for both the prediction and constraining of, important input parameters, generating annual modulation predictions for these experiments. 

Dark Mark defines a class, Nibbler, which contains all pertinent information regarding both the astrophysical inputs, and detector characteristics like halo model, velocity distribution of dark matter, detector materials, dark matter mass and cross-section. Dark Mark uses Pynbody_ to sample dark matter from the input simulation, or accepts an analytical model of the distribution of dark matter, like the Maxwell Boltzmann speed distribution. It then calculates the differential spectral rate function, :math:`\frac{dR}{dE_R}`, which will demonstrate the expected event rate at a given day of the year, over a range of energies. Users have the option to specify an energy window of interest, in order to generate annual modulation curve predictions. 


Dark Mark can be installed via the source code here_.

.. _here: https://github.com/Grace-Lawrence/Dark-MaRK
.. _Pynbody: https://pynbody.github.io/pynbody/

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/introduction
   user_guide/installation
   user_guide/methods
   user_guide/guidelines



