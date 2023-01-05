import sys
sys.path.append("/home/sven/repos/gaia-tools/gaia_tools")

USE_CUDA=True

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

FINAL_DATA_COLUMNS = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'r', 'phi', 'v_r', 'v_phi',
            'sig_vphi', 'sig_vr', 'source_id']


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

def get_galcen_data(r_0):

   # Update solar vector
   v_sun[1][0] = 251.5*(r_0/8277)
   v_sun[2][0] = 8.59*(r_0/8277)

   galcen_data = transformation_functions.get_transformed_data(icrs_data,
                                       include_cylindrical = True,
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       v_sun = v_sun,
                                       is_bayes = True,
                                       NUMPY_LIB = npcp,
                                       dtype = dtype)

   # ["ra", "dec","r_est","pmra","pmdec","radial_velocity"] -> [:,1::]
   galactocentric_cov = cov.transform_cov_matrix(C = C_icrs, 
                                       df = icrs_data[:,1::],
                                       coordinate_system = 'Cartesian',
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       is_bayes = True,
                                       NUMPY_LIB = npcp,
                                       dtype = dtype)

   # ["x", "y","r","phi","v_r","v_phi"] -> [0,1,6,7,8,9]
   cyl_cov_gpu = cov.transform_cov_matrix(C = galactocentric_cov, 
                                       df = galcen_data[:,[0,1,6,7,8,9]],
                                       coordinate_system = 'Cylindrical',
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       is_bayes = False,
                                       NUMPY_LIB = npcp,
                                       dtype = dtype)

   sig_vphi = npcp.array([cyl_cov_gpu[:,4,4]])
   sig_vr = npcp.array([cyl_cov_gpu[:,3,3]])
   source_id = npcp.array([icrs_data[:,0]])
   galcen_data = npcp.concatenate(([galcen_data, sig_vphi.T, sig_vr.T, source_id.T]), axis=1)

   return galcen_data


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


# Fully vectorised
def bootstrap_weighted_error_gpu_vector(bin_vphi, bin_sig_vphi):
    
    num_it = 1000
    data_length = len(bin_vphi)
    idx_list = npcp.arange(data_length)
    bootstrapped_means = npcp.zeros(num_it)

    rnd_idx = npcp.random.choice(idx_list, replace=True, size=(num_it, data_length))
    
    test_sample = bin_vphi[rnd_idx]
    sig_vphi = bin_sig_vphi[rnd_idx]
    bootstrapped_means = (test_sample/sig_vphi).sum(axis=1)/(1/sig_vphi).sum(axis=1)
    conf_int = npcp.percentile(bootstrapped_means, [16, 84])

    return (conf_int[1] - conf_int [0])/2

debug = False

def log_likelihood(theta, args):

   if(debug):
      tic=timeit.default_timer()

   h_r = theta[-3]
   h_sig = theta[-2]
   r_0 = theta[-1]

   with DeviceContext(device_id):

      # Get Galactocentric data
      galcen_data = get_galcen_data(r_0)

      # Turn Galactocentric data into Pandas frame
      galcen_data = pd.DataFrame(galcen_data.get(), columns=FINAL_DATA_COLUMNS)
      
      r_min = 5000
      r_max = 15000

      # # Generate bins
      bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                            theta = (0, 1),
                                                            BL_r_min = r_min,
                                                            BL_r_max = r_max,
                                                            BL_z_min = -200,
                                                            BL_z_max = 200,
                                                            N_bins = (args.nbins, 1),
                                                            r_drift = False,
                                                            debug = False)

      n = reduce(lambda x, y: x*y, bin_collection.N_bins)
      likelihood_array = np.zeros(n)

      for i, bin in enumerate(bin_collection.bins):
         bin.bootstrapped_error = bootstrap_weighted_error_gpu_vector(npcp.asarray(bin.data.v_phi, dtype=dtype), 
                                                            npcp.asarray(bin.data.sig_vphi, dtype=dtype))
         bin.A_parameter = bin.compute_A_parameter(h_r = h_r, 
                                                h_sig = h_sig, 
                                                debug=False)

         likelihood_value = bin.get_likelihood_w_asymmetry(theta[i], drop_approx=True, debug=False)

         likelihood_array[i] = likelihood_value
   likelihood_sum = np.sum(likelihood_array)

   if(debug):
       toc=timeit.default_timer()
       print("time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

   return likelihood_sum

def log_prior(theta, args):

   vc_prior_d = (theta[0:-3] > -400).all()
   vc_prior_u = (theta[0:-3] < 400).all()

   disk_prior = (theta[-3] > args.disk_scale - 1000) and (theta[-3] < args.disk_scale + 1000)
   vlos_prior = (theta[-2] > args.vlos_dispersion_scale - 1000) and (theta[-2] < args.vlos_dispersion_scale + 1000)

   r0_prior = (theta[-1] > 7800 and theta[-1] < 8500)

   if vc_prior_d and vc_prior_u and disk_prior and vlos_prior and r0_prior:
         return 0.0
   return -np.inf

def log_probability(theta, args):

   lp = log_prior(theta, args)
   if not np.isfinite(lp):
       return -np.inf
   return lp + log_likelihood(theta, args)

def init_vars(queue, input_data, input_cov):
   
   trans_needed_columns = ['source_id', 'ra', 'dec', 'r_est', 'pmra', 'pmdec', 'radial_velocity']
   global device_id
   global icrs_data
   global C_icrs
   global z_0
   global v_sun
   
   device_id = queue.get()

   with DeviceContext(device_id):
      icrs_data = npcp.array(input_data[trans_needed_columns], dtype=dtype)
      C_icrs = npcp.array(input_cov, dtype=dtype)

      z_0 = 25

      # Initial solar vector
      import transformation_constants
      v_sun = transformation_constants.V_SUN
      v_sun[0][0] = 11.1
      v_sun[1][0] = 251.5
      v_sun[2][0] = 8.59

if __name__ == '__main__':

   multiprocessing.set_start_method('forkserver')

   args = parse_args()

   dtime = dt.time()
   now=dt.datetime.now()
   start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")

   print('Creating outpath for current run...')
   run_out_path = "/home/sven/repos/gaia-tools/out/mcmc_runs/{}_{}".format(start_datetime, args.nwalkers)
   Path(run_out_path).mkdir(parents=True, exist_ok=True)

   print('Importing necessary column names...')
   icrs_data_columns = pd.read_csv("/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv", nrows = 10).columns

   print('Importing DR3...')
   dr3_path = '/local/mariacst/2022_v0_project/data/GaiaDR3_RV_RGB_fidelity.csv'
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

   r_0, z_0, v_sun = load_galactic_parameters()

   # Nwalkers has to be at least 2*ndim
   nwalkers = args.nwalkers
   ndim = args.nbins + 3
   nsteps = args.nsteps
   theta_0 = random.sample(range(-300, -200), ndim)

   theta_0[-3] = args.disk_scale
   theta_0[-2] = args.vlos_dispersion_scale
   theta_0[-1] = r_0

   # Init starting point for all walkers
   pos = theta_0 + 10**(-1)*np.random.randn(nwalkers, ndim)

   # Setup saving results to output file
   filename = run_out_path + "/sampler_{a}.h5".format(a=start_datetime)
   backend = emcee.backends.HDFBackend(filename)
   backend.reset(nwalkers, ndim)

   if USE_CUDA: 
      cvd = os.environ["CUDA_VISIBLE_DEVICES"]
      cvd = [int(x) for x in cvd.split(",")]
      NUM_GPUS = len(cvd)
      print("Num GPUs {}".format(NUM_GPUS))
      print("Total num CPUs {}".format(multiprocessing.cpu_count()))
   
   else:
      NUM_GPUS = 1
 
   PROC_PER_GPU = 2
   print("Using {} CPUs per GPU".format(PROC_PER_GPU))

   queue = Queue()
   #even though CUDA_VISIBLE_DEVICES could be e.g. 3,4
   #here the indexing will be from 0,1, as nvidia hides the other devices
   for gpu_ids in range(NUM_GPUS):
      for _ in range(PROC_PER_GPU):
         queue.put(gpu_ids)

   # Setup pool   
   pool = multiprocessing.Pool(NUM_GPUS*PROC_PER_GPU, initializer=init_vars, initargs=(queue, icrs_data,C_icrs,))

   # Run emcee EnsembleSampler
   with pool as pool:
      sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, args=(args,), backend=backend)
      print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, NUM_GPUS*PROC_PER_GPU))
      sampler.run_mcmc(pos, nsteps, progress=True)
      print("Sampler done!")

   pool.close()
   pool.join()

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

   # To continue previous run
   # filename = "/home/svenpoder/repos/gaia-tools/out/mcmc_runs/2022-12-02-16-09-40_range0.3/sampler_2022-12-02-16-09-40.h5"
   # reader = emcee.backends.HDFBackend(filename)
   # samples = reader.get_chain()
   # pos = samples[-1]