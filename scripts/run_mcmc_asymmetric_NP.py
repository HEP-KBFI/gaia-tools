
'''
MCMC to run in the cluster for: R0, U_odot, V_odot
'''
import sys
sys.path.append("../gaia_tools/")
import data_analysis
import covariance_generation as cov
from import_functions import import_data
import numpy as np
import emcee
from functools import reduce
import time, timeit
import transformation_constants
import datetime as dt
import pandas as pd
import os
import pickle
import argparse


# 2MASS DATA IMPORT
tmass_data_path = '/hdfs/local/sven/gaia_tools_data/crossmatched_tmass_data/crossmatched_tmass_data.csv'

import re
def is_allowed_flg(string):

    charRe = re.compile(r'[^A-B]')
    string = charRe.search(string)
    return not bool(string)

crossmatched_tmass_data = pd.read_csv(tmass_data_path)

sliced_tmass_data = crossmatched_tmass_data[((crossmatched_tmass_data.j_m - crossmatched_tmass_data.k_m) > 0.5) & ((crossmatched_tmass_data.j_m - crossmatched_tmass_data.k_m) < 1.1)]
sliced_tmass_data = sliced_tmass_data[((sliced_tmass_data.j_msigcom) < 0.1) & ((sliced_tmass_data.h_msigcom) < 0.1) & ((sliced_tmass_data.k_msigcom) < 0.1)]
sliced_tmass_data =  sliced_tmass_data [sliced_tmass_data.apply(lambda row : is_allowed_flg(row.ph_qual), axis=1) == True]
sliced_tmass_data.reset_index(inplace=True, drop=True)


# GAIA DATA IMPORT

# The path containing the initial ICRS data with Bayesian distance estimates.
my_path = "/hdfs/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv"

# Writing new import section for faster debugging!
start = time.time()

# Import the ICRS data
icrs_data = import_data(path = my_path, is_bayes = True, debug = True)


# MERGE 2MASS GAIA ON 'source_id'
icrs_data = icrs_data.merge(sliced_tmass_data, on='source_id', suffixes=('', '_tmass'))[icrs_data.columns]
print('Gaia data shape: {}'.format(icrs_data.shape))

# Use stars within 1.5 kpc of the Sun
# r_est_lim = 1500
# icrs_data = icrs_data[(icrs_data.r_est < r_est_lim)]
# icrs_data.reset_index(inplace=True, drop=True)

z_lim = 50
r_lim_min = 5000
r_lim_max = 12000


galcen_data = data_analysis.get_transformed_data(icrs_data,
                                       include_cylindrical = True,
                                       debug = True,
                                       is_bayes = True,
                                       is_source_included = True)

galcen_data = galcen_data[(galcen_data.r < r_lim_max) & (galcen_data.r > r_lim_min)]
galcen_data = galcen_data[(galcen_data.z < z_lim) & (galcen_data.z > -z_lim)]
galcen_data.reset_index(inplace=True, drop=True)

icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]
print('Gaia data shape: {}'.format(icrs_data.shape))

coords =  data_analysis.transform_coordinates_galactocentric(icrs_data,
                                        transformation_constants.Z_0,
                                        transformation_constants.R_0,
                                        True)
phi = np.arctan2(coords[:,1],coords[:,0])
CYL_MAT = transformation_constants.get_cylindrical_velocity_matrix(phi)

galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, True)


## VELOCITY TRANSFORMATION SETUP

n = len(icrs_data)

# Going from DEG -> RAD
ra = np.deg2rad(icrs_data.ra)
dec = np.deg2rad(icrs_data.dec)

# from 1/yr -> km/s
k2 = transformation_constants.k2

# Assign r estiamtes to c2
c2 = icrs_data.r_est
c2 = k2*(icrs_data.r_est/1000)

# Initial velocity vector in ICRS in units km/s
v_ICRS = np.array([[icrs_data.radial_velocity],
                    [(c2)*icrs_data.pmra],
                    [(c2)*icrs_data.pmdec]])

v_ICRS = v_ICRS.T.reshape(n,3,1, order = 'A')

B = transformation_constants.get_b_matrix(ra.to_numpy(), dec.to_numpy())
B = B.reshape(n,3,3, order = 'A')

# Using M1, M2, M3, .. for transparency in case of bugs
M1 = B @ v_ICRS
M2 = transformation_constants.A @ M1
M3 = transformation_constants.get_H_matrix(transformation_constants.Z_0, transformation_constants.R_0) @ M2

def transform_velocities_galactocentric(M3, v_sun):

    # Return is a np.array of shape (n,3,1)
    M4 = M3 + v_sun

    return M4

def transform_velocities_cylindrical(velocities_xyz, CYL_MAT):
    v_cylindrical = CYL_MAT @ velocities_xyz
    return v_cylindrical


#######################################
# End import section

debug = False

dtime = dt.time()
now=dt.datetime.now()
start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

def log_likelihood(theta):

   # transform data
   # region

   '''
   transfrom from icrs to cylindrical galactocentric coordinates

   '''

   v_sun = np.array([[transformation_constants.V_SUN[0][0]],
                     [theta[-1]],
                     [transformation_constants.V_SUN[2][0]]])

   temp_galcen_data = galcen_data

   # TRANSFORM ONLY VELOCITIES - VSUN NOT ENTER COORDINATE TRANSFORM
   velocities = transform_velocities_galactocentric(M3, v_sun)
   vel_cyl = transform_velocities_cylindrical(velocities, CYL_MAT)

   velocities_df = pd.DataFrame(np.squeeze(velocities, axis=2), columns="v_x v_y v_z".split())
   cyl_velocities_df = pd.DataFrame(np.squeeze(vel_cyl[:,0:2], axis=2), columns="v_r v_phi".split())
   velocities_df = pd.concat([velocities_df, cyl_velocities_df], axis=1)

   temp_galcen_data[velocities_df.columns] = velocities_df

   cyl_cov = cov.transform_cov_cylindirical(temp_galcen_data, galactocentric_cov)
   temp_galcen_data = temp_galcen_data.merge(cyl_cov, on='source_id')

   #endregion

   # bin data
   #region

   min_val = np.min(temp_galcen_data.r)
   max_val = np.max(temp_galcen_data.r)

   # declared variable with bincollection object
   bin_collection = data_analysis.get_collapsed_bins(data = temp_galcen_data,
                                                      theta = (theta[0], theta[1]),
                                                      BL_r_min = min_val - 1,
                                                      BL_r_max = max_val + 1,
                                                      BL_z_min = -z_lim,
                                                      BL_z_max = z_lim,
                                                      N_bins = (5, 1),
                                                      r_drift = False,
                                                      debug = False)

   # populates bins with their mle values of mean and variance
   # bin_collection.GetMLEParameters()

   #endregion

   for bin in bin_collection.bins:
      bin.med_sig_vphi = np.median(bin.data.sig_vphi)
      bin.A_parameter = bin.compute_A_parameter()

   if(debug):
      tic=timeit.default_timer()

   # calculate likelihoods region
   n = reduce(lambda x, y: x*y, bin_collection.N_bins)
   likelihood_array = np.zeros(n)

   # now we need to calculate likelihood values for each bin
   for i, bin in enumerate(bin_collection.bins):

      # get bin likelihood
      likelihood_value = bin.get_likelihood_w_asymmetry(theta[i], debug=False)

      likelihood_array[i] = likelihood_value

   likelihood_sum = np.sum(likelihood_array)

   if(debug):
       toc=timeit.default_timer()
       print("time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

   #endregion

   return likelihood_sum

def log_prior(theta):

   ## prior assumptions of our parameters
   ## flat across all parameters at first

   trig_1 = (theta[0:-1] > -400).all() and (theta[0:-1]  < 300).all()
   trig_2 = (theta[-1] > 150) and (theta[-1]  < 300)

   if trig_1 and trig_2:
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
nwalkers = 60
ndim = 6
nsteps = 1500

# These are randomly chosen in the prior range
theta_0 = (-300, -190, -210, -275, -147, 200)

# Init starting point for all walkers
pos = theta_0 + 10**(-3)*np.random.randn(nwalkers, ndim)

print(pos)

# Setup saving results to output file
filename = "../out/mcmc_sampler/bovy_sample/sampler_{}_zlim{}_minrlim{}_maxrlim{}.h5".format(start_datetime, z_lim, r_lim_min, r_lim_max)

# To continue previous run

#prev_filename = "/home/sven/repos/gaia-tools/out/mcmc_sampler/modified_binning/sampler_larger_binning_10x1_2022-05-17-11-49-12_zlim50_minrlim5000_maxrlim12000.h5"
# reader = emcee.backends.HDFBackend(filename)
# samples = reader.get_chain()
# pos = samples[-1]

backend = emcee.backends.HDFBackend(filename)

# Comment out to continue previous run
#backend.reset(nwalkers, ndim)

with Pool(ncpu) as pool:

   # Init emcee EnsembleSampler
   sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend=backend)

   print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, ncpu))
   # Run the sampler
   sampler.run_mcmc(pos, nsteps, progress=True)

   print("Sampler done!")

# Dumps sampler object to pkl
fn='../out/mcmc_sampler/sampler_pkls/sampler_bovy_sample_5x1_optimized_nsteps{}_zlim{}_minrlim{}_maxrlim{}'.format(nsteps,
                                                   z_lim, 
                                                   r_lim_min, r_lim_max)
with open(os.path.splitext(fn)[0] + ".pkl", "wb") as f:
        pickle.dump(sampler, f, -1)


print("Script finished!")

#return result