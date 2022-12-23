
import sys
sys.path.append("../gaia_tools/")

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
from multiprocessing import Pool
import pandas as pd 


z_0 = 25

def get_galcen_data(z_0, r_0, v_sun):

   galcen_data = transformation_functions.get_transformed_data(icrs_data,
                                       include_cylindrical = True,
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       v_sun = v_sun,
                                       debug = True,
                                       is_bayes = True,
                                       is_source_included = True, 
                                       NUMPY_LIB = NUMPY_LIB,
                                       dtype = dtype)

   # ["ra", "dec","r_est","pmra","pmdec","radial_velocity"] -> [:,1::]
   galactocentric_cov_gpu = cov.transform_cov_matrix(C = C_icrs, 
                                       df = icrs_data[:,1::],
                                       coordinate_system = 'Cartesian',
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       is_bayes = True,
                                       NUMPY_LIB = NUMPY_LIB,
                                       dtype = dtype)

   # ["x", "y","r","phi","v_r","v_phi"] -> [0,1,6,7,8,9]
   cyl_cov_gpu = cov.transform_cov_matrix(C = galactocentric_cov_gpu, 
                                       df = galcen_data[:,[0,1,6,7,8,9]],
                                       coordinate_system = 'Cylindrical',
                                       z_0 = z_0,
                                       r_0 = r_0,
                                       is_bayes = False,
                                       NUMPY_LIB = NUMPY_LIB,
                                       dtype = dtype)

   sig_vphi = cp.array([cyl_cov_gpu[:,4,4]])
   sig_vr = cp.array([cyl_cov_gpu[:,3,3]])
   source_id = cp.array([icrs_data[:,0]])
   galcen_data = cp.concatenate(([galcen_data, sig_vphi.T, sig_vr.T, source_id.T]), axis=1)

   return galcen_data


   # galcen_data = galcen_data.merge(cyl_cov, on='source_id')

   # # Final data selection
   # galcen_data = galcen_data[(galcen_data.r < 15000) & (galcen_data.r > 5000)]
   # galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
   # galcen_data.reset_index(inplace=True, drop=True)
   
   # return galcen_data



# End import and plot section
debug = False

def log_likelihood(theta, v_sun):

   if(debug):
      tic=timeit.default_timer()

   h_r = theta[-3]
   h_sig = theta[-2]
   r_0 = theta[-1]

   # Update solar vector
   v_sun[1][0] = 251.5*(r_0/8275)
   v_sun[2][0] = 8.59*(r_0/8275)

   # Get Galactocentric data as Numpy array
   galcen_data = get_galcen_data(z_0, r_0, v_sun)

   # Turn Galactocentric data into Pandas frame
   galcen_data = pd.DataFrame(galcen_data.get(), columns=final_data_columns)
   
   # # Generate bins
   bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                         theta = (0, 1),
                                                         BL_r_min = 2500,
                                                         BL_r_max = 12500,
                                                         BL_z_min = -200,
                                                         BL_z_max = 200,
                                                         N_bins = (args.nbins, 1),
                                                         r_drift = False,
                                                         debug = False)

   n = reduce(lambda x, y: x*y, bin_collection.N_bins)
   likelihood_array = np.zeros(n)

   # now we need to calculate likelihood values for each bin
   for i, bin in enumerate(bin_collection.bins):
      bin.bootstrapped_error = helpfunc.bootstrap_weighted_error(bin.data.v_phi.to_numpy(), bin.data.sig_vphi.to_numpy())
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
   #z0_prior = (theta[-1] > -4 and theta[-1] < 30)

   if vc_prior_d and vc_prior_u and disk_prior and vlos_prior and r0_prior:
         return 0.0
   return -np.inf

def log_probability(theta, args, v_sun):

   lp = log_prior(theta, args)
   if not np.isfinite(lp):
       return -np.inf
   return lp + log_likelihood(theta, v_sun)



if __name__ == '__main__':

   # Arguments
   parser = argparse.ArgumentParser(description='MCMC input')
   parser.add_argument('--nwalkers', type=int)
   parser.add_argument('--nsteps', type=int)
   parser.add_argument('--nbins', type=int)
   parser.add_argument('--disk-scale', type=float)
   parser.add_argument('--vlos-dispersion-scale', type=float)
   parser.add_argument('--backend', type=str)
   args = parser.parse_args()

   if(args.backend == 'gpu'):

      print('Setting backend as GPU...')

      import cupy as cp
      from numba import jit, config
      config.DISABLE_JIT = True
      NUMPY_LIB = cp
      dtype = cp.float32
   else:
      import numpy
      NUMPY_LIB = numpy
      dtype = numpy.float64

   import transformation_constants
   import transformation_functions
   import data_analysis
   import covariance_generation as cov
   from import_functions import import_data
   from data_plot import sample_distribution_galactic_coords, plot_radial_distribution, plot_distribution, display_polar_histogram, plot_variance_distribution, plot_velocity_distribution

   dtime = dt.time()
   now=dt.datetime.now()
   start_datetime = now.strftime("%Y-%m-%d-%H-%M-%S")


   print('Grabbing needed columns')
   icrs_data = pd.read_csv('/home/svenpoder/DATA/Gaia_2MASS Data_DR2/gaia_rv_data_bayes.csv', nrows = 10)

   print('Importing DR3')
   path = '/home/svenpoder/DATA/Gaia_DR3/GaiaDR3_RV_RGB_fidelity.csv'
   gaia_dr3 = pd.read_csv(path, nrows=1e5)
   icrs_data = gaia_dr3[icrs_data.columns]
   print("Initial size of sample: {}".format(icrs_data.shape))

   # Create outpath for current run
   run_out_path = "../out/mcmc_runs/{}_{}".format(start_datetime, args.nwalkers)
   Path(run_out_path).mkdir(parents=True, exist_ok=True)

   ## TRANSFORMATION CONSTANTS
   v_sun = transformation_constants.V_SUN

   # Initial Galactocentric distance
   r_0 = 8277

   # Initial height over Galactic plane
   z_0 = 25

   # Initial solar vector
   v_sun[0][0] = 11.1
   v_sun[1][0] = 251.5*(r_0/8275)
   v_sun[2][0] = 8.59*(r_0/8275)

   ## TRANSFORM DATA USING INITIAL TRANSFORMATION CONSTANTS
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

   # Selection plots
   plot_distribution(galcen_data, run_out_path, 'r', 0, 20000, [5000, 15000])
   plot_distribution(galcen_data, run_out_path, 'z', -2000, 2000, [-200, 200])

   # Final data selection
   galcen_data = galcen_data[(galcen_data.r < 15000) & (galcen_data.r > 5000)]
   galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
   galcen_data.reset_index(inplace=True, drop=True)
   icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]
   print("Final size of sample {}".format(galcen_data.shape))

   # Generate covariance matrix in ICRS
   C_icrs = cov.generate_covmat(icrs_data)

   # Sample distribution plots
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

   # Plots the velocity and velocity variance distribution of first 4 bins. 
   plot_velocity_distribution(bin_collection.bins[0:4], run_out_path, True)
   plot_variance_distribution(bin_collection.bins[0:4], 'v_phi', run_out_path)


   # GPU VARIABLES
   if(args.backend == 'gpu'):
      trans_needed_columns = ['source_id', 'ra', 'dec', 'r_est', 'pmra', 'pmdec', 'radial_velocity']
      icrs_data = cp.asarray(icrs_data[trans_needed_columns], dtype=cp.float32)
      C_icrs = cp.asarray(C_icrs, dtype=cp.float32)

   final_data_columns = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'r', 'phi', 'v_r', 'v_phi',
         'sig_vphi', 'sig_vr', 'source_id']

   # Define CPU count
   ncpu = 6
   print("{0} CPUs".format(ncpu))

   # Nwalkers has to be at least 2*ndim
   nwalkers = args.nwalkers
   ndim = args.nbins + 3
   nsteps = args.nsteps
   theta_0 = random.sample(range(-300, -150), ndim)

   theta_0[-3] = args.disk_scale
   theta_0[-2] = args.vlos_dispersion_scale
   theta_0[-1] = r_0

   # Init starting point for all walkers
   pos = theta_0 + 10**(-3)*np.random.randn(nwalkers, ndim)

   # Setup saving results to output file
   filename = run_out_path + "/sampler_{a}.h5".format(a=start_datetime)

   # To continue previous run
   # filename = "/home/svenpoder/repos/gaia-tools/out/mcmc_runs/2022-12-02-16-09-40_range0.3/sampler_2022-12-02-16-09-40.h5"
   # reader = emcee.backends.HDFBackend(filename)
   # samples = reader.get_chain()
   # pos = samples[-1]

   backend = emcee.backends.HDFBackend(filename)
   backend.reset(nwalkers, ndim)

   multiprocessing.set_start_method('spawn', force=True)

   # Run emcee EnsembleSampler
   with Pool(ncpu) as pool:
      sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool, args=(args,v_sun),  backend=backend)
      print("Starting sampling. Walkers = {}, Steps = {}, CPU = {}".format(nwalkers, nsteps, ncpu))
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
               'cut_range' : args.cut_range,
               'final_sample_size' : galcen_data.shape,
               'disk_scale' : args.disk_scale,
               'vlos_dispersion_scale' : args.vlos_dispersion_scale,
               'A_r_info' : A_r_array}

   pickle.dump(out_dict, file)
   file.close()

   print("Script finished!")
      