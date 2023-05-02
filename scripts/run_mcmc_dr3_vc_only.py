
import sys
sys.path.append("../gaia_tools/")
import data_analysis
import covariance_generation as cov
import numpy as np
import emcee
from functools import reduce
import time, timeit
import transformation_constants
import datetime as dt
import os
import pickle
from pathlib import Path
import argparse
import random
import pandas as pd
import helper_functions as helpfunc 

def load_galactic_parameters():
   
   # # Initial Galactocentric distance
   r_0 = 8277

   # # Initial height over Galactic plane
   z_0 = 25

   # # Initial solar vector
   v_sun = transformation_constants.V_SUN
   v_sun[0][0] = 11.1
   v_sun[1][0] = 251.5*(r_0/8277)
   v_sun[2][0] = 8.59*(r_0/8277)

   # -----------------------------------
   # Referee check
   # z_0 = 0

   # Eilers et al orbital parmaeters
   # r_0 = 8122
   # z_0 = 25
   # v_sun = transformation_constants.V_SUN
   # v_sun[0][0] = 11.1
   # v_sun[1][0] = 245.8
   # v_sun[2][0] = 7.8
   
   return r_0, z_0, v_sun

def apply_initial_cut(icrs_data, run_out_path):

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

   galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, 
                                                               is_bayes = True,
                                                               Z_0 = z_0,
                                                               R_0 = r_0)

   cyl_cov = cov.transform_cov_cylindirical(galcen_data, 
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



dtime = dt.time()
now=dt.datetime.now()
start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

# Arguments
parser = argparse.ArgumentParser(description='MCMC input')
parser.add_argument('--cut-range', type=float)
parser.add_argument('--nwalkers', type=int)
parser.add_argument('--nsteps', type=int)
parser.add_argument('--nbins', type=int)
parser.add_argument('--disk-scale', type=float)
parser.add_argument('--vlos-dispersion-scale', type=float)
args = parser.parse_args()


print('Creating outpath for current run...')
custom_ext = 'VC_ONLY_default_params'
run_out_path = "../out/mcmc_runs/{}_{}_{}".format(start_datetime, args.nwalkers, custom_ext)
Path(run_out_path).mkdir(parents=True, exist_ok=True)

print('Importing DR3...')
dr3_path = '/local/sven/v0_project_archive/GaiaDR3_RV_RGB_fidelity.csv'
#dr3_path = '/home/svenpoder/DATA/Gaia_DR3/GaiaDR3_RV_RGB_fidelity.csv'
#dr3_path = '/storage/users/benitoca/2022_v0_project/data/GaiaDR3/GaiaDR3_RV_RGB_fidelity.csv'
gaia_dr3 = pd.read_csv(dr3_path)

r_est_error = (gaia_dr3.B_rpgeo - gaia_dr3.b_rpgeo)/2
gaia_dr3['r_est_error'] = r_est_error

columns_to_drop = ['Vbroad', 'GRVSmag', 'Gal', 'Teff', 'logg',
      '[Fe/H]', 'Dist', 'A0', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
      'RADEcorJ2000', 'B_Teff', 'b_Teff', 'b_logg', 'B_logg', 'b_Dist',
      'B_Dist', 'b_AG', 'B_AG', 'b_A0', 'B_A0', 'Gmag', 'BPmag', 'RPmag', 'BP-RP']
gaia_dr3 = gaia_dr3.drop(columns=columns_to_drop)
print(gaia_dr3.columns)
icrs_data = gaia_dr3

parallax_over_error = icrs_data.parallax/icrs_data.parallax_error
icrs_data['parallax_over_error'] = parallax_over_error

print("Initial size of sample: {}".format(icrs_data.shape))


print('Applying cut...')
galcen_data = apply_initial_cut(icrs_data, run_out_path)
print("Final size of sample {}".format(galcen_data.shape))

# Declare final sample ICRS data and covariance matrices
icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]
C_icrs = cov.generate_covmat(icrs_data)

r_0, z_0, v_sun = load_galactic_parameters()
r_min = 5000/r_0
r_max = 15000/r_0

# Generate bins
bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                      theta = r_0,
                                                      BL_r_min = r_min,
                                                      BL_r_max = r_max,
                                                      BL_z_min = -200,
                                                      BL_z_max = 200,
                                                      N_bins = (args.nbins, 1),
                                                      r_drift = True,
                                                      debug = False)

# A parameter computation
for i, bin in enumerate(bin_collection.bins):
    bin.bootstrapped_error = helpfunc.bootstrap_weighted_error(bin.data.v_phi.to_numpy(), bin.data.sig_vphi.to_numpy())
    bin.A_parameter = bin.compute_A_parameter(h_r = args.disk_scale,
                                             h_sig = args.vlos_dispersion_scale,
                                             debug=False)

# End import and plot section

debug = False

def log_likelihood(theta):

   if(debug):
      tic=timeit.default_timer()

   n = reduce(lambda x, y: x*y, bin_collection.N_bins)
   likelihood_array = np.zeros(n)

   # now we need to calculate likelihood values for each bin
   for i, bin in enumerate(bin_collection.bins):
      likelihood_value = bin.get_likelihood_w_asymmetry(theta[i], drop_approx=True, debug=False)
      likelihood_array[i] = likelihood_value
   likelihood_sum = np.sum(likelihood_array)

   if(debug):
       toc=timeit.default_timer()
       print("time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

   return likelihood_sum

def log_prior(theta):

   # NOTE CHANGE BACK
   if (theta > -400).all() and (theta < 400).all():
       return 0.0
   return -np.inf

def log_probability(theta):

   lp = log_prior(theta)
   if not np.isfinite(lp):
       return -np.inf
   return lp + log_likelihood(theta)

from multiprocessing import Pool
from multiprocessing import cpu_count

# Define CPU count
ncpu = 6
print("{0} CPUs".format(ncpu))

# Nwalkers has to be at least 2*ndim
nwalkers = args.nwalkers
ndim = args.nbins
nsteps = args.nsteps
theta_0 = random.sample(range(-300, -150), ndim)

# Init starting point for all walkers
pos = theta_0 + 10**(-3)*np.random.randn(nwalkers, ndim)

# Setup saving results to output file
filename = run_out_path + "/sampler_{a}.h5".format(a=start_datetime)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Run emcee EnsembleSampler
with Pool(ncpu) as pool:
   sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend=backend)
   print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, ncpu))
   sampler.run_mcmc(pos, nsteps, progress=True)
   print("Sampler done!")

#Dumps sampler object to pkl
fn = run_out_path + '/sampler_{}'.format(start_datetime)
with open(os.path.splitext(fn)[0] + ".pkl", "wb") as f:
   pickle.dump(sampler, f, -1)

bin_centers_r = [np.mean(x.r_boundaries) for x in bin_collection.bins]
bin_centers_z = [np.mean(x.z_boundaries) for x in bin_collection.bins]

A_r_array = []
for i, bin in enumerate(bin_collection.bins):
   A_r_array.append((np.mean(bin.r_boundaries), bin.A_parameter))

file = open(run_out_path + "/run_settings.txt", "wb")
out_dict = {'bin_centers_r' : np.array(bin_centers_r),
            'bin_centers_z' : np.array(bin_centers_z),
            'bin_edges' : bin_collection.bin_boundaries,
            'nbins' : args.nbins,
            'V_sun' : v_sun,
            'R_0' : r_0,
            'Z_0' : z_0,
            'cut_range' : args.cut_range,
            'final_sample_size' : galcen_data.shape,
            'disk_scale' : args.disk_scale,
            'vlos_dispersion_scale' : args.vlos_dispersion_scale,
            'A_r_info' : A_r_array}

pickle.dump(out_dict, file)
file.close()

print("Script finished!")

