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
    
    def sub_sum():

        a = np.array([v_i[i]*denom_sum(i) for i in range(n)]).sum()

        b = np.array([denom_sum(i) for i in range(n)]).sum()

        return a, b


    def denom_sum(i):

        sum_component = 1/(sigma**2 + s_i[i]**2)

        return sum_component
    
    a, b = sub_sum()
    
    result_1 = np.array([ ((v_i[i] - a/b)**2)/((sigma**2 + s_i[i]**2))**2 for i in range(n)]).sum()
    
    result_2 = b
    
    return -result_1 + result_2



def get_MLE_sigma(v_i, s_i):

    out = minimize(objective_function, params, args=(v_i, s_i))

    assert out.success == True, "Bin MLE variance could not be found!"
    assert out.params['sigma'].value >= 0, "Bin MLE variance can't be negative!"
        
    return out.params['sigma'].value


def get_MLE_mu(sigma, v_i, s_i):
    
    n = len(v_i)
    
    def denom_sum(i):

        sum_component = 1/(sigma**2 + s_i[i]**2)

        return sum_component
    
    a = np.array([v_i[i]*denom_sum(i) for i in range(n)]).sum()

    b = np.array([denom_sum(i) for i in range(n)]).sum()
    
    return a/b