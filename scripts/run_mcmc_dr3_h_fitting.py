import sys
sys.path.append("../gaia_tools/")

USE_CUDA=False

if USE_CUDA:
   import cupy as npcp
   DeviceContext = npcp.cuda.Device
   dtype = npcp.float32
   from numba import config
   config.DISABLE_JIT = True

else:
   import numpy as npcp
   npcp.asnumpy = lambda x: x
   import contextlib
   DeviceContext = contextlib.nullcontext
   dtype = npcp.float64
   from numba import config
   config.DISABLE_JIT = False

import transformation_constants
import transformation_functions
import helper_functions as helpfunc
import data_analysis
import covariance_generation as cov
from import_functions import import_data
from data_plot import sample_distribution_galactic_coords, plot_radial_distribution, plot_distribution, display_polar_histogram, plot_variance_distribution, plot_velocity_distribution
import numpy as np
import emcee
from functools import reduce
import time, timeit
import datetime as dt
import os
import pickle
from pathlib import Path
import argparse
import random
import multiprocessing
from multiprocessing import Pool, Process, Queue
import pandas as pd 


def parse_args():
   parser = argparse.ArgumentParser(description='MCMC input')
   parser.add_argument('--nwalkers', type=int)
   parser.add_argument('--nsteps', type=int)
   parser.add_argument('--nbins', type=int)
   parser.add_argument('--disk-scale', type=float)
   parser.add_argument('--vlos-dispersion-scale', type=float)
   parser.add_argument('--backend', type=str)
   return parser.parse_args()

def load_galactic_parameters():
   
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

   galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, 
                                                               is_bayes = True,
                                                               Z_0 = z_0,
                                                               R_0 = r_0)

   cyl_cov = cov.transform_cov_cylindirical(galcen_data, 
                                                C = galactocentric_cov,
                                                Z_0 = z_0,
                                                R_0 = r_0)
   galcen_data = galcen_data.merge(cyl_cov, on='source_id')

   # Plots before cut
   plot_distribution(galcen_data, run_out_path, 'r', 0, 20000, [5000, 15000])
   plot_distribution(galcen_data, run_out_path, 'z', -2000, 2000, [-200, 200])

   # Final data cut
   galcen_data = galcen_data[(galcen_data.r < 15000) & (galcen_data.r > 5000)]
   galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
   galcen_data.reset_index(inplace=True, drop=True)
   
   return galcen_data

debug = False

def bootstrap_weighted_error_gpu(bin_vphi, bin_sig_vphi):
    
    total_num_it = 1000
    batch_num = 10
    data_length = len(bin_vphi)
    idx_list = npcp.arange(data_length)
    bootstrapped_means = npcp.zeros(total_num_it)

    for i in range(100):
        rnd_idx = npcp.random.choice(idx_list, replace=True, size=(batch_num, data_length))
        
        test_sample = bin_vphi[rnd_idx]
        sig_vphi = bin_sig_vphi[rnd_idx]

        start_idx = (i+1)*batch_num - batch_num
        end_idx = (i+1)*batch_num

        bootstrapped_means[start_idx:end_idx] = (test_sample/sig_vphi).sum(axis=1)/(1/sig_vphi).sum(axis=1)
    conf_int = npcp.percentile(bootstrapped_means, [16, 84])

    return (conf_int[1] - conf_int [0])/2

def log_likelihood(theta, args):

   if(debug):
      tic=timeit.default_timer()

   likelihood_array = np.zeros(args.nbins)

   # now we need to calculate likelihood values for each bin
   for i, bin in enumerate(bin_collection.bins):
      bin.A_parameter = bin.compute_A_parameter(h_r = theta[-2], 
                                             h_sig = theta[-1], 
                                             debug=False)

      likelihood_value = bin.get_likelihood_w_asymmetry(theta[i], drop_approx=True, debug=False)
      likelihood_array[i] = likelihood_value
   likelihood_sum = np.sum(likelihood_array)

   if(debug):
       toc=timeit.default_timer()
       print("time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

   return likelihood_sum

def log_prior(theta, args):

   vc_prior_d = (theta[0:-2] > -400).all()
   vc_prior_u = (theta[0:-2] < 400).all()

   disk_prior = (theta[-2] > args.disk_scale - 1000) and (theta[-2] < args.disk_scale + 1000)
   vlos_prior = (theta[-1] > args.vlos_dispersion_scale - 1000) and (theta[-1] < args.vlos_dispersion_scale + 1000)

   if vc_prior_d and vc_prior_u and disk_prior and vlos_prior:
       return 0.0
   return -np.inf

def log_probability(theta, args):

   lp = log_prior(theta, args)
   if not np.isfinite(lp):
       return -np.inf
   return lp + log_likelihood(theta, args)

def init_vars(queue, bins):
   
   global device_id
   global bin_collection
   
   device_id = queue.get()

   with DeviceContext(device_id):
      bin_collection = bins

if __name__ == '__main__':

   multiprocessing.set_start_method('forkserver')

   args = parse_args()

   dtime = dt.time()
   now=dt.datetime.now()
   start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

   print('Creating outpath for current run...')
   custom_ext = 'H_FITTING_TEST'
   run_out_path = "../out/mcmc_runs/{}_{}_{}".format(start_datetime, args.nwalkers, custom_ext)
   Path(run_out_path).mkdir(parents=True, exist_ok=True)

   print('Importing necessary column names...')
   icrs_data_columns = pd.read_csv('/home/svenpoder/DATA/Gaia_2MASS Data_DR2/gaia_rv_data_bayes.csv', nrows = 10).columns

   print('Importing DR3...')
   dr3_path = '/home/svenpoder/DATA/Gaia_DR3/GaiaDR3_RV_RGB_fidelity.csv'
   gaia_dr3 = pd.read_csv(dr3_path)
   icrs_data = gaia_dr3[icrs_data_columns]
   print("Initial size of sample: {}".format(icrs_data.shape))

   print('Applying cut...')
   galcen_data = apply_initial_cut(icrs_data, run_out_path)
   galcen_data = galcen_data[::10]
   print("Final size of sample {}".format(galcen_data.shape))
   
   # Declare final sample ICRS data and covariance matrices
   icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]
   C_icrs = cov.generate_covmat(icrs_data)

   # Plots after cut
   sample_distribution_galactic_coords(icrs_data, run_out_path)
   plot_radial_distribution(icrs_data, run_out_path)
   fig2 = display_polar_histogram(galcen_data, run_out_path, r_limits=(0, 15000), norm_max=5000, title = "Distribution of data on the Galactic plane")

   # Generate bins
   bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                         theta = (0, 1),
                                                         BL_r_min = 5000,
                                                         BL_r_max = 15000,
                                                         BL_z_min = -200,
                                                         BL_z_max = 200,
                                                         N_bins = (args.nbins, 1),
                                                         r_drift = False,
                                                         debug = False)

   # Bootstrap errors
   for i, bin in enumerate(bin_collection.bins):
      if(USE_CUDA):
         bin.bootstrapped_error = bootstrap_weighted_error_gpu(npcp.asarray(bin.data.v_phi, dtype=dtype), 
                                                               npcp.asarray(bin.data.sig_vphi, dtype=dtype))
      else:
         bin.bootstrapped_error = helpfunc.bootstrap_weighted_error(bin.data.v_phi.to_numpy(), bin.data.sig_vphi.to_numpy())
   
   r_0, z_0, v_sun = load_galactic_parameters()

   # SETUP MCMC
   # Nwalkers has to be at least 2*ndim
   nwalkers = args.nwalkers
   ndim = args.nbins + 2
   nsteps = args.nsteps
   theta_0 = random.sample(range(-300, -200), ndim)

   theta_0[-2] = args.disk_scale
   theta_0[-1] = args.vlos_dispersion_scale

   # Init starting point for all walkers
   pos = theta_0 + 10**(-1)*np.random.randn(nwalkers, ndim)

   # Setup saving results to output file
   filename = run_out_path + "/sampler_{a}.h5".format(a=start_datetime)
   backend = emcee.backends.HDFBackend(filename)
   backend.reset(nwalkers, ndim)

   # if USE_CUDA: 
   #    cvd = os.environ["CUDA_VISIBLE_DEVICES"]
   #    cvd = [int(x) for x in cvd.split(",")]
   #    NUM_GPUS = len(cvd)
    #actually no GPUs will be used, we just create 1xPROC_PER_GPU CPU processes
   # else:
   NUM_GPUS = 1
 
   PROC_PER_GPU = 16
   queue = Queue()
   #even though CUDA_VISIBLE_DEVICES could be e.g. 3,4
   #here the indexing will be from 0,1, as nvidia hides the other devices
   for gpu_ids in range(NUM_GPUS):
      for _ in range(PROC_PER_GPU):
         queue.put(gpu_ids)

   # Setup pool   
   pool = multiprocessing.Pool(NUM_GPUS*PROC_PER_GPU, initializer=init_vars, initargs=(queue, bin_collection))

   # Run emcee EnsembleSampler
   with pool as pool:
      sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, args=(args,), backend=backend)
      print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, NUM_GPUS*PROC_PER_GPU))
      sampler.run_mcmc(pos, nsteps, progress=True)
      print("Sampler done!")

   #Dumps sampler object to pkl
   fn = run_out_path + '/sampler_{}'.format(start_datetime)
   with open(os.path.splitext(fn)[0] + ".pkl", "wb") as f:
      pickle.dump(sampler, f, -1)

   bin_centers_r = [np.mean(x.r_boundaries) for x in bin_collection.bins]
   bin_centers_z = [np.mean(x.z_boundaries) for x in bin_collection.bins]

   # A parameter computation for the asymmetric drift plots
   for i, bin in enumerate(bin_collection.bins):
      bin.A_parameter = bin.compute_A_parameter(h_r = args.disk_scale, 
                                                h_sig = args.vlos_dispersion_scale, 
                                                debug=True)

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
               'final_sample_size' : galcen_data.shape,
               'disk_scale' : args.disk_scale,
               'vlos_dispersion_scale' : args.vlos_dispersion_scale,
               'A_r_info' : A_r_array}

   pickle.dump(out_dict, file)
   file.close()

   print("Script finished!")
