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

    bins = data_analysis.get_collapsed_bins(galcen_data, BL_r = 100000, BL_z = 5000, N_bins = (10, 10))

    #endregion

    #region Calculate Likelihoods
    
    likelihood_array = np.zeros(nbins)

    for bin in bins:

        # Calculate some MLE value inside bin
        likelihood_value = 0;

        np.append(likelihood_array, likelihood_value)
    
    # this goes to a sum 
    likelihood_product = np.prod(likelihood_array)

    #endregion
    return likelihood_product

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
