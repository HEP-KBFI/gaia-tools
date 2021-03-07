'''
Separate module for the numerical solver.
'''

from scipy.optimize import fsolve
import math
from sympy import *
from numpy import random
import numpy as np

# Here I declare all the neccessary symbols
s = symbols('s')
mu = symbols('mu')

# i-th component of given velocity array
v_i = symbols('v')

# i-th component of given measurement error array
s_i = symbols('serr')

# Summation integer
i = symbols('i', integer=True)

# First equation (MLE of mu)
a = Function('a')

# Second equation (MLE of sigma)
b = Function('b')

def a(v_arr, s_arr):
    
    sum1 = Sum((Indexed('v',i)/(s**2 + Indexed('serr',i)**2)), (i, 0, n-1)) - mu*Sum((1/(s**2 + Indexed('serr',i)**2)), (i, 0, n-1))
    
    f1 = lambdify([v_i, s_i], sum1)
    
    return f1(v_arr, s_arr)


def b(v_arr, s_arr):
    
    sum2 = Sum((1/(s**2 + Indexed('serr',i)**2)), (i, 0, n-1)) - Sum(((Indexed('v',i) - mu)**2/(s**2 + Indexed('serr',i)**2)**2), (i, 0, n-1))
    
    f2 = lambdify([v_i, s_i], sum2)
    
    f2(v_test, s_test)
    
    return f2(v_arr, s_arr)

def get_bin_MLE(v_arr, s_arr):

    res = nsolve((a(v_arr, s_arr),b(v_arr, s_arr)), 
                 (s, mu), 
                 (10,10), 
                 dict=True)

    return res