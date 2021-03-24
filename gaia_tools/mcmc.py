'''
Class containing neccessary functions for MCMC loop.
'''
from . import data_analysis
from . import covariance_generation as cov
import numpy as np
import emcee
from functools import reduce
import time, timeit

#def run_sampler_multicore(theta_0 = None, nwalkers = None, ndim = None, steps = 10, processes = 4, icrs_data = None):

#    def log_likelihood(theta):

#        # Transform Data
#        #region 

#        '''
#        Transfrom from ICRS to Cylindrical galactocentric coordinates 

#        '''
    
#        v_sun = (theta[2], theta[3], theta[4])

#        galcen_data = data_analysis.get_transformed_data(icrs_data, 
#                                                            include_cylindrical = True, 
#                                                            z_0 = theta[1],
#                                                            r_0 = theta[0],
#                                                            v_sun = v_sun,
#                                                            debug = False, 
#                                                            is_source_included = True)
    
#        cov_df = cov.generate_covmatrices(df = icrs_data, 
#                                            df_crt = galcen_data, 
#                                            transform_to_galcen = True, 
#                                            transform_to_cylindrical = True,
#                                            z_0 = theta[1],
#                                            r_0 = theta[0],
#                                            debug=False)

#        # Append covariance information to galactocentric data
#        galcen_data['cov_mat'] = cov_df['cov_mat']

#        #endregion 

#        # Bin Data
#        #region

#        # Declared variable with BinCollection object
#        bin_collection = data_analysis.get_collapsed_bins(galcen_data, BL_r = 100000, BL_z = 5000, N_bins = (10, 10), debug = False)

#        # Populates bins with their MLE values of mean and variance
#        bin_collection.GetMLEParameters()

#        #endregion

#        if(debug):
#            tic=timeit.default_timer()

#        # Calculate Likelihoods
#        #region 
#        n = reduce(lambda x, y: x*y, bin_collection.N_bins)
#        likelihood_array = np.zeros(n)

#        # Now we need to calculate likelihood values for each bin
#        for i, bin in enumerate(bin_collection.bins):

#            # Get Bin likelihood
#            likelihood_value = bin.get_bin_likelihood()
        
#            # Add to array
#            likelihood_array[i] = likelihood_value
    
#        # Square sum likelihoods over all bins
#        likelihood_sum = np.sum(likelihood_array**2)

#        if(debug):
#            toc=timeit.default_timer()
#            print("Time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

#        #endregion

#        return likelihood_sum

#    def log_prior(theta):

#        ## Prior assumptions of our parameters
#        ## Flat across all parameters at first

#        r, z, u_sun, v_sun, w_sun = theta
#        if 0.0 < r < 10000.0 and 0.0 < z < 100.0 and 0.0 < u_sun < 100.0 and 0.0 < v_sun < 1000.0 and 0.0 < w_sun < 100.0:
#            return 0.0

#        return -np.inf


#    def log_probability(theta):

#        lp = log_prior(theta)

#        if not np.isfinite(lp):
#            return -np.inf

#        return lp + log_likelihood(theta)





#    import os

#    from multiprocessing import Pool
#    from multiprocessing import cpu_count

#    ncpu = cpu_count()
#    print("{0} CPUs".format(ncpu))
        
#    nwalkers = nwalkers
#    ndim = ndim
#    # Init starting point for all walkers
#    pos = theta_0 + 1e-4 * np.random.randn(nwalkers, ndim)

#    with Pool(processes) as pool:


#        # Init emcee EnsembleSampler
#        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool = pool)

#        # Run the sampler
#        sampler.run_mcmc(pos, steps, progress=True);

#        print("Sampler done!")

#        result = sampler

#        return result

## TODO: Configure burn in steps!
#def run_sampler(theta_0 = None, nwalkers = None, ndim = None,steps = 10):

#    nwalkers = nwalkers
#    ndim = ndim

#    # Init starting point for all walkers
#    pos = theta_0 + 1e-4 * np.random.randn(nwalkers, ndim)

#    # Init emcee EnsembleSampler
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

#    # Run the sampler
#    sampler.run_mcmc(pos, steps, progress=True);

#    print("Sampler done!")

#    result = sampler

#    return True

#def drop_burn_in(result=None, discard = 100, thin = 15, flat=True):

#    flat_result = result.get_chain(discard = discard, thin = thin, flat = flat)

#    return flat_result



'''
OLD CODE WITH CLASS STRUCTURE

Couldnt get it to work with multiprocessing

'''

class MCMCLooper:

    def __init__(self, icrs_data, theta_0, nwalkers=16, ndim=5, debug = False):

        self.icrs_data = icrs_data
        self.theta_0 = theta_0
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.debug = debug
        self.iter_step = 0
        self.result = None
        
    def log_likelihood(self, theta):

        # Transform Data
        #region 

        '''
        Transfrom from ICRS to Cylindrical galactocentric coordinates 

        '''
    
        v_sun = (theta[2], theta[3], theta[4] )

        galcen_data = data_analysis.get_transformed_data(self.icrs_data, 
                                                         include_cylindrical = True, 
                                                         z_0 = theta[1],
                                                         r_0 = theta[0],
                                                         v_sun = v_sun,
                                                         debug = self.debug, 
                                                         is_source_included = True)
    
        cov_df = cov.generate_covmatrices(df = self.icrs_data, 
                                            df_crt = galcen_data, 
                                            transform_to_galcen = True, 
                                            transform_to_cylindrical = True,
                                            z_0 = theta[1],
                                            r_0 = theta[0],
                                            debug=self.debug)

        # Append covariance information to galactocentric data
        galcen_data['cov_mat'] = cov_df['cov_mat']

        #endregion 

        # Bin Data
        #region

        # Declared variable with BinCollection object
        bin_collection = data_analysis.get_collapsed_bins(galcen_data, BL_r = 100000, BL_z = 5000, N_bins = (10, 10), debug = self.debug)

        # Populates bins with their MLE values of mean and variance
        bin_collection.GetMLEParameters()

        #endregion

        if(self.debug):
            tic=timeit.default_timer()

        # Calculate Likelihoods
        #region 
        n = reduce(lambda x, y: x*y, bin_collection.N_bins)
        likelihood_array = np.zeros(n)

        # Now we need to calculate likelihood values for each bin
        for i, bin in enumerate(bin_collection.bins):

            # Get Bin likelihood
            likelihood_value = bin.get_bin_likelihood()
        
            # Add to array
            likelihood_array[i] = likelihood_value
    
        # Square sum likelihoods over all bins
        likelihood_sum = np.sum(likelihood_array**2)

        if(self.debug):
            toc=timeit.default_timer()
            print("Time elapsed for likelihoods computation section: {a} sec".format(a=toc-tic))

        #endregion

        return likelihood_sum

    def log_prior(self, theta):

        ## Prior assumptions of our parameters
        ## Flat across all parameters at first

        r, z, u_sun, v_sun, w_sun = theta
        if 0.0 < r < 10000.0 and 0.0 < z < 100.0 and 0.0 < u_sun < 100.0 and 0.0 < v_sun < 1000.0 and 0.0 < w_sun < 100.0:
            return 0.0

        return -np.inf


    def log_probability(self, theta):

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta)

    # TODO: Configure burn in steps!
    def run_sampler(self, steps = 10):

        nwalkers = self.nwalkers
        ndim = self.ndim

        # Init starting point for all walkers
        pos = self.theta_0 + 1e-4 * np.random.randn(nwalkers, ndim)

        # Init emcee EnsembleSampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)

        # Run the sampler
        sampler.run_mcmc(pos, steps, progress=True);

        print("Sampler done!")

        self.result = sampler

        return True

    def drop_burn_in(self, discard = 100, thin = 15, flat=True):

        flat_result = self.result.get_chain(discard = discard, thin = thin, flat = flat)

        return flat_result


    def run_sampler_multicore(self, steps = 10, processes = 4):
        import os

        from multiprocessing import Pool
        from multiprocessing import cpu_count

        ncpu = cpu_count()
        print("{0} CPUs".format(ncpu))
        
        nwalkers = self.nwalkers
        ndim = self.ndim
        # Init starting point for all walkers
        pos = self.theta_0 + 1e-4 * np.random.randn(nwalkers, ndim)

        with Pool(processes) as pool:


            # Init emcee EnsembleSampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool = pool)

            # Run the sampler
            sampler.run_mcmc(pos, steps, progress=True);

            print("Sampler done!")

            self.result = sampler

            return True


