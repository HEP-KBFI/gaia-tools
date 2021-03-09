'''
Class containing neccessary functions for MCMC loop.
'''
from . import data_analysis
from . import covariance_generation as cov
import numpy as np
import emcee

class MCMCLooper:

    def __init__(self, icrs_data, theta_0, debug = False):

        self.icrs_data = icrs_data
        self.theta_0 = theta_0
        self.debug = debug

   

    def log_likelihood(theta):


        # Example likelihood from emcee documentation
        #m, b, log_f = theta
        #model = m * x + b
        #sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
        #return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        # -----------------------------------------------

        #region Transform Data

        '''
        Transfrom from ICRS to Cylindrical galactocentric coordinates 

        '''
    
        galcen_data = data_analysis.get_transformed_data(self.data_icrs, 
                                                         include_cylindrical = True, 
                                                         z_0 = theta.z,
                                                         r_0 = theta.r,
                                                         v_sun = theta.v_sun,
                                                         debug = True, 
                                                         is_source_included = True)
    
        cov_df = cov.generate_covmatrices(df = self.data_icrs, 
                                            df_crt = galcen_data, 
                                            transform_to_galcen = False, 
                                            transform_to_cylindrical = True,
                                            z_0 = theta.z,
                                            r_0 = theta.r,
                                            debug=True)

        # Append covariance information to galactocentric data
        galcen_data['cov_mat'] = cov_df['cov_mat']

        #endregion 

        #region Bin Data

        # Declared variable with BinCollection object
        bins = data_analysis.get_collapsed_bins(galcen_data, BL_r = 100000, BL_z = 5000, N_bins = (10, 10))

        # Populates bins with their MLE values of mean and variance
        bins.GetMLEParameters()

        #endregion

        #region Calculate Likelihoods
    
        likelihood_array = np.zeros(bins.N_bins)

        # Now we need to calculate likelihood values for each bin
        for i, bin in enumerate(bins):

            # Get Bin likelihood
            likelihood_value = bin.get_bin_likelihood()
        
            # Add to array
            likelihood_array[i] = likelihood_value
    
        # Square sum likelihoods over all bins
        likelihood_sum = np.sum(likelihood_array**2)

        #endregion

        return likelihood_sum

    def log_prior(theta):

        ## Prior assumptions of our parameters
        ## Flat across all parameters at first

        r, z, u_sun, v_sun, w_sun = theta
        if 0.0 < r < 10000.0 and 0.0 < z < 100.0 and 0.0 < u_sun < 100.0 and 0.0 < v_sun < 1000.0 and 0.0 < w_sun < 100.0:
            return 0.0

        return -np.inf


    def log_probability(theta):

        lp = log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + log_likelihood(theta)

    # TODO: Configure burn in steps!
    def run_sampler():

        nwalkers = 32
        ndim = 5

        # Init starting point for all walkers
        pos = self.theta_0 + 1e-4 * np.random.randn(nwalkers, ndim)

        # Init emcee EnsembleSampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

        # Run the sampler
        sampler.run_mcmc(pos, 5000, progress=True);

        print("Sampler done!")

        return sampler
