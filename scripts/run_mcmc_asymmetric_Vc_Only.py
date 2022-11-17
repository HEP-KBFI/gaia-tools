
import sys
sys.path.append("../gaia_tools/")
import data_analysis
import covariance_generation as cov
from import_functions import import_data
from data_plot import sample_distribution_galactic_coords, plot_radial_distribution, plot_distribution, display_polar_histogram, plot_variance_distribution, plot_velocity_distribution
import numpy as np
import emcee
from functools import reduce
import time, timeit
import transformation_constants
import datetime as dt
import photometric_cut
import os
import pickle
from pathlib import Path
import argparse
import random

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

# Create outpath for current run
run_out_path = "../out/mcmc_runs/{}_range{}".format(start_datetime, args.cut_range)
Path(run_out_path).mkdir(parents=True, exist_ok=True)

print("Photometric cut..")
sample_IDs = photometric_cut.get_sample_IDs(run_out_path, args.cut_range, True)

# The path containing the initial ICRS data with Bayesian distance estimates.
my_path = "/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv"

# Import ICRS data
icrs_data = import_data(path = my_path, is_bayes = True, debug = True)
icrs_data = icrs_data.merge(sample_IDs, on='source_id', suffixes=("", "_y"))
icrs_data.reset_index(inplace=True, drop=True)

print("Size of sample after diagonal cut in ROI {}".format(icrs_data.shape))

## TRANSFORMATION CONSTANTS
v_sun = transformation_constants.V_SUN

#Eilers et al.
v_sun[0][0] = 11.1
v_sun[1][0] = 245.8
v_sun[2][0] = 7.8
z_0 = 25
r_0 = 8122

# z_0 = transformation_constants.Z_0
# r_0 = transformation_constants.R_0

galcen_data = data_analysis.get_transformed_data(icrs_data,
                                       include_cylindrical = True,
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       v_sun = v_sun,
                                       debug = True,
                                       is_bayes = True,
                                       is_source_included = True)

galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, True)
cyl_cov = cov.transform_cov_cylindirical(galcen_data, galactocentric_cov)
galcen_data = galcen_data.merge(cyl_cov, on='source_id')

# Selection plots
plot_distribution(galcen_data, run_out_path, 'r', 0, 20000, [5000, 12000])
plot_distribution(galcen_data, run_out_path, 'z', -2000, 2000, [-200, 200])

# Final data selection
galcen_data = galcen_data[(galcen_data.r < 12000) & (galcen_data.r > 5000)]
galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
galcen_data.reset_index(inplace=True, drop=True)
print("Final size of sample {}".format(galcen_data.shape))

icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]

# Sample distribution plots
sample_distribution_galactic_coords(icrs_data, run_out_path)
plot_radial_distribution(icrs_data, run_out_path)
fig2 = display_polar_histogram(galcen_data, run_out_path, r_limits=(0, 15000), norm_max=5000, title = "Distribution of data on the Galactic plane")

min_r = np.min(galcen_data.r)
max_r = np.max(galcen_data.r)

# Generate bins
bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                      theta = (0, 1),
                                                      BL_r_min = min_r - 1,
                                                      BL_r_max = max_r + 1,
                                                      BL_z_min = -200,
                                                      BL_z_max = 200,
                                                      N_bins = (args.nbins, 1),
                                                      r_drift = False,
                                                      debug = False)

# Plots the velocity and velocity variance distribution of first 4 bins.
plot_velocity_distribution(bin_collection.bins[0:4], run_out_path, True)
plot_variance_distribution(bin_collection.bins[0:4], 'v_phi', run_out_path)

# A parameter computation
for i, bin in enumerate(bin_collection.bins):
    bin.med_sig_vphi = np.median(bin.data.sig_vphi)
    bin.A_parameter = bin.compute_A_parameter(h_r = args.disk_scale,
                                             h_sig = args.vlos_dispersion_scale,
                                             debug=True)

# End import and plot section

debug = False

def log_likelihood(theta):

   if(debug):
      tic=timeit.default_timer()

   n = reduce(lambda x, y: x*y, bin_collection.N_bins)
   likelihood_array = np.zeros(n)

   # now we need to calculate likelihood values for each bin
   for i, bin in enumerate(bin_collection.bins):
      likelihood_value = bin.get_likelihood_w_asymmetry(theta[i], debug=False)
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

