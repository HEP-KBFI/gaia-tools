
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
import photometric_cut
import os
import pickle

dtime = dt.time()
now=dt.datetime.now()
start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

print("Starting MCMC..")

sample_IDs = photometric_cut.get_sample_IDs(start_datetime, True)

# The path containing the initial ICRS data with Bayesian distance estimates.
my_path = "/hdfs/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv"

# Import ICRS data
icrs_data = import_data(path = my_path, is_bayes = True, debug = True)
icrs_data = icrs_data.merge(sample_IDs, on='source_id', suffixes=("", "_y"))
icrs_data.reset_index(inplace=True, drop=True)

print("Size of sample after diagonal cut in ROI {}".format(icrs_data.shape))

galcen_data = data_analysis.get_transformed_data(icrs_data,
                                       include_cylindrical = True,
                                       debug = True,
                                       is_bayes = True,
                                       is_source_included = True)

galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, True)
cyl_cov = cov.transform_cov_cylindirical(galcen_data, galactocentric_cov)
galcen_data = galcen_data.merge(cyl_cov, on='source_id')

# append covariance information to galactocentric data
#galcen_data['cov_mat'] = cov_df['cov_mat']

galcen_data = galcen_data[(galcen_data.r < 12000) & (galcen_data.r > 5000)]
galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
galcen_data.reset_index(inplace=True, drop=True)

print("Final size of sample {}".format(galcen_data.shape))

icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]

min_val = np.min(galcen_data.r)
max_val = np.max(galcen_data.r)

bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                      theta = (0, 1),
                                                      BL_r_min = min_val - 1,
                                                      BL_r_max = max_val + 1,
                                                      BL_z_min = -200,
                                                      BL_z_max = 200,
                                                      N_bins = (5, 1),
                                                      r_drift = False,
                                                      debug = False)

for i, bin in enumerate(bin_collection.bins):
    bin.med_sig_vphi = np.median(bin.data.sig_vphi)
    bin.A_parameter = bin.compute_A_parameter()

# End import section

debug = False

def log_likelihood(theta):

   if(debug):
      tic=timeit.default_timer()

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

   return likelihood_sum

def log_prior(theta):

   ## prior assumptions of our parameters
   ## flat across all parameters at first

   if (theta > -400).all() and (theta< -100).all():
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
nwalkers = 50
ndim = 5
nsteps = 1500


theta_0 = (-300,-190, -210, -275, -147)


# Init starting point for all walkers
pos = theta_0 + 10**(-3)*np.random.randn(nwalkers, ndim)

print(pos)

# Setup saving results to output file
filename = "../out/mcmc_sampler/sampler_{a}.h5".format(a=start_datetime)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

with Pool(ncpu) as pool:

   # Init emcee EnsembleSampler
   sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, backend=backend)

   print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, ncpu))
   # Run the sampler
   sampler.run_mcmc(pos, nsteps, progress=True)

   print("Sampler done!")

# Dumps sampler object to pkl
fn='../out/mcmc_sampler/sampler_pkls/sampler_{}'.format(start_datetime)
with open(os.path.splitext(fn)[0] + ".pkl", "wb") as f:
        pickle.dump(sampler, f, -1)


print("Script finished!")

#return result