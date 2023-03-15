import sys
sys.path.append("../gaia_tools/")
sys.path.append("../scripts/")
import covariance_generation
import transformation_constants
import data_analysis
from mcmc_plots import *
import numpy as np
import emcee
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from pylab import *
from scipy.optimize import curve_fit
from cProfile import label
import pandas as pd

dr3_path = '/local/mariacst/2022_v0_project/data/GaiaDR3_RV_RGB_fidelity.csv'
gaia_dr3 = pd.read_csv(dr3_path)

r_est_error = (gaia_dr3.B_rpgeo - gaia_dr3.b_rpgeo)/2
gaia_dr3['r_est_error'] = r_est_error

columns_to_drop = ['Unnamed: 0', 'Vbroad', 'GRVSmag', 'Gal', 'Teff', 'logg',
       '[Fe/H]', 'Dist', 'A0', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
       'RADEcorJ2000', 'B_Teff', 'b_Teff', 'b_logg', 'B_logg', 'b_Dist',
       'B_Dist', 'b_AG', 'B_AG', 'b_A0', 'B_A0', 'Gmag', 'BPmag', 'RPmag', 'BP-RP']
gaia_dr3 = gaia_dr3.drop(columns=columns_to_drop)

parallax_over_error = gaia_dr3.parallax/gaia_dr3.parallax_error
gaia_dr3['parallax_over_error'] = parallax_over_error

print(gaia_dr3.columns)

def load_galactic_parameters():
   '''The load_galactic_parameters function sets the initial galactocentric distance, height over the Galactic plane, and solar vector used in the coordinate transformations.'''

   # Initial Galactocentric distance
   r_0 = 8277

   # Initial height over Galactic plane
   z_0 = 25

   # Initial solar vector
   v_sun = transformation_constants.V_SUN
   v_sun[0][0] = 11.1
   v_sun[1][0] = 251.5
   v_sun[2][0] = 8.59

   return r_0, z_0, v_sun

def apply_initial_cut(icrs_data):

   r_0, z_0, v_sun = load_galactic_parameters()

   galcen_data = data_analysis.get_transformed_data(icrs_data,
                                          include_cylindrical = True,
                                          z_0 = z_0,
                                          r_0 = r_0,
                                          v_sun = v_sun,
                                          debug = True,
                                          is_bayes = True,
                                          is_source_included = True)

   print("Galactocentric data shape: {}".format(galcen_data.shape))

   galactocentric_cov = covariance_generation.generate_galactocentric_covmat(icrs_data, 
                                                               is_bayes = True,
                                                               Z_0 = z_0,
                                                               R_0 = r_0)

   cyl_cov = covariance_generation.transform_cov_cylindirical(galcen_data, 
                                                C = galactocentric_cov,
                                                Z_0 = z_0,
                                                R_0 = r_0)
   galcen_data = galcen_data.merge(cyl_cov, on='source_id')

   print("Galactocentric data shape after merge with covariance info: {}".format(galcen_data.shape))

   # Remove noisy distances
   print("Removing noisy distances")
   galcen_data['parallax_over_error'] = icrs_data.parallax_over_error[galcen_data.source_id == icrs_data.source_id]
   galcen_data = galcen_data[galcen_data.parallax_over_error > 5]
   galcen_data = galcen_data.drop(columns=['parallax_over_error'])

   print("Galactocentric data shape after removing noisy distances: {}".format(galcen_data.shape))

   # Final data cut
   galcen_data = galcen_data[(galcen_data.r < 15000) & (galcen_data.r > 5000)]
   galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]

   print("Galactocentric data shape after constraining region: {}".format(galcen_data.shape))

   # Remove halo stars (condition taken from 1806.06038)                        
   v_dif = np.linalg.norm(np.array([galcen_data.v_x, galcen_data.v_y, galcen_data.v_z])-v_sun,
                        axis=0)                                               
   galcen_data['v_dif'] = v_dif                                                 
   galcen_data = galcen_data[galcen_data.v_dif<210.]

   print("Galactocentric data shape after removing halo stars: {}".format(galcen_data.shape))

   galcen_data.reset_index(inplace=True, drop=True)
   
   return galcen_data

galcen_data = apply_initial_cut(gaia_dr3)

print(galcen_data.shape)

gaia_dr3 = gaia_dr3.merge(galcen_data, on='source_id')[gaia_dr3.columns]

print('Dumping to file.')
gaia_dr3.to_csv('/local/sven/v0_project_archive/Poder_vc_DR3_input.csv', index=False)