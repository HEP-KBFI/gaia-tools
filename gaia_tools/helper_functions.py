import numpy as np
import cupy as cp
from numba import jit

@jit(nopython=True)
def bootstrap_weighted_error(bin_vphi, bin_sig_vphi):
    
    data_length = len(bin_vphi)
    idx_list = np.arange(data_length)
    bootstrapped_means = np.zeros(1000)

    for i in range(1000):
        rnd_idx = np.random.choice(idx_list, replace=True, size=data_length)
        test_sample = bin_vphi[rnd_idx]
        sig_vphi = bin_sig_vphi[rnd_idx]
        bootstrapped_means[i] = (test_sample/sig_vphi).sum()/(1/sig_vphi).sum()
    conf_int = np.percentile(bootstrapped_means, [16, 84])

    return (conf_int[1] - conf_int [0])/2


def bootstrap_weighted_error_gpu(bin_vphi, bin_sig_vphi):
    
    total_num_it = 1000
    batch_num = 10
    data_length = len(bin_vphi)
    idx_list = cp.arange(data_length)
    bootstrapped_means = cp.zeros(total_num_it)

    for i in range(100):
        rnd_idx = cp.random.choice(idx_list, replace=True, size=(batch_num, data_length))
        
        test_sample = bin_vphi[rnd_idx]
        sig_vphi = bin_sig_vphi[rnd_idx]

        start_idx = (i+1)*batch_num - batch_num
        end_idx = (i+1)*batch_num

        bootstrapped_means[start_idx:end_idx] = (test_sample/sig_vphi).sum(axis=1)/(1/sig_vphi).sum(axis=1)
    conf_int = cp.percentile(bootstrapped_means, [16, 84])

    return (conf_int[1] - conf_int [0])/2


