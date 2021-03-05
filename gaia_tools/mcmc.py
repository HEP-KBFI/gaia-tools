'''
Module containing neccessary functions for MCMC loop.
'''

import numpy as np
import emcee

# LogLikelihood for each bin?
# If so will pass in arguments transformed Bin object

def log_likelihood(theta, data_icrs):
    #m, b, log_f = theta
    #model = m * x + b
    #sigma2 = yerr ** 2 + model ** 2 * np.exp(2 * log_f)
    #return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
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
    pos = THETA_0

    nwalkers = 32
    ndim = 5

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data_icrs))
    sampler.run_mcmc(pos, 5000, progress=True);

    print("Sampler done!")

    return sampler
