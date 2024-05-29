# Homepage

## Introduction

This repository contains code to help with general Gaia data handling and to perform a Bayesian analysis of the kinematics of stars in the Milky Way as described in A Bayesian Estimation of the Milky Way’s Circular Velocity Curve using Gaia DR3 (Põder, Benito et al. 2023).

The code is all written in python3 and makes substantial use of the standard pandas, numpy, matplotlib, etc. Additional libraries that you may need to investigate depending on your installation:

* [astropy](https://www.astropy.org/)
* [emcee](https://emcee.readthedocs.io/en/stable/)
* [lmfit](https://lmfit.github.io/lmfit-py/examples/example_brute.html)
* [corner](https://corner.readthedocs.io/en/latest/)
* [h5py](https://www.h5py.org/)
* [numba](http://numba.pydata.org/)

The input data used in the paper can be found on [Zenodo.](https://zenodo.org/record/7755721#.ZBw3QI5BxQp)

All Gaia parameters can be found [here.](https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html) 

## A brief summary of Gaia

The Gaia Space Telescope was launched in December of 2013 and reached its planned orbit, at the L2 Lagrange Point about 1.5 million km from the Earth, within three weeks. 
The Gaia mission is lead by the European Space Agency (ESA) and its main goal is to map the most accurate and voluminous three-dimensional map of our Galaxy. 
The spacecraft itself is designed to maximise the accuracy, sensitivity, dynamic range and sky coverage to a level which has not yet been achieved. 

Gaia's structure consists of three modules: a payload module, which contains all the necessary instruments for the observation and data processing side of the mission, a mechanical service module and an electrical service module, both of which are keeping the spacecraft operational and have for example a thermal tent, sunshield, solar arrays etc. 
Observations are based on spectra and take place in the range of 320 - 1000 nm.

The data of the stars is measured by five astrometric parameters: two angular position coordinates, two proper motions components and the trigonometric parallax. 
Furthermore, the radial-velocity spectrometer (RVS), measures the radial velocities of the stellar objects through Doppler-shifts using cross-correlation for stars brighter than G = 16 mag.

## What does this Documentation contain?

The first topic of this Documentation is 'Galactocentric_transformation' which basically means transforming the data we get from the Gaia DR3 to other coordinate systems, in this case from ICRS to Galactocentric Cartesian and then to the Galactocentric Cylindrical coordinate system.

In 'Error propagation' we transform to the same coordinate systems as previously, but now with the error ranges of the coordinates and velocities. 

And lastly 'CPU,GPU Optimization' ...