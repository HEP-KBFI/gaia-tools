'''
Module containing neccessary functions for MCMC loop.
'''
from . import data_analysis
from . import covariance_generation as cov
import numpy as np
import emcee

# LogLikelihood for all the bins
# If so will pass in arguments transformed Bin object

def log_likelihood(theta, data_icrs):


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
    
    galcen_data = data_analysis.get_transformed_data(data_icrs, 
                                                     include_cylindrical = True, 
                                                     z_0 = theta.z,
                                                     r_0 = theta.r,
                                                     v_sun = theta.v_sun,
                                                     debug = True, 
                                                     is_source_included = True)
    
    cov_dict = cov.generate_covmatrices(df = data_icrs, 
                                        df_crt = galcen, 
                                        transform_to_galcen = False, 
                                        transform_to_cylindrical = True,
                                        z_0 = theta.z,
                                        r_0 = theta.r,
                                        debug=True)

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

    return -np.inf


def log_probability(theta, data_icrs):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, data_icrs)


def run_sampler(THETA_0):

    # Our initial starting point for the sampler
    # Fix this
    pos = THETA_0

    nwalkers = 32
    ndim = 5

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, 5000, progress=True);

    print("Sampler done!")

    return sampler
