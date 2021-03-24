'''
Separate module for the numerical solver.
'''

import numpy as np
from lmfit import minimize, Parameters

params = Parameters()

guess = 10
params.add('sigma', value = guess, min = 0, max = 1000)
params

def objective_function(params, v_i, s_i):
    
    sigma = params['sigma']
    
    n = len(v_i)
    
    denom_array = (sigma**2 + s_i**2)**(-1)

    a = (v_i*denom_array).sum()
    b = denom_array.sum()

    result_1 = (((v_i-a/b)**2)*(denom_array)**2).sum()

    result_2 = b

    return -result_1 + result_2



def get_MLE_sigma(v_i, s_i):

    out = minimize(objective_function, params, args=(v_i, s_i))

    assert out.success == True, "Bin MLE variance could not be found!"
    assert out.params['sigma'].value >= 0, "Bin MLE variance can't be negative!"
        
    return out.params['sigma'].value


def get_MLE_mu(sigma, v_i, s_i):
    
    n = len(v_i)
    
    denom_array = (sigma**2 + s_i**2)**(-1)

    a = (v_i*denom_array).sum()
    b = denom_array.sum()
    
    return a/b