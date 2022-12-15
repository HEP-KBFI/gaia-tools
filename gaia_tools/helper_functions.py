import numpy as np
from numba import jit

@jit(nopython=True)
def bootstrap_weighted_error(bin_vphi, bin_sig_vphi):
    
    data_length = len(bin_vphi)
    idx_list = np.arange(data_length)
    bootstrapped_means = np.zeros(100)

    for i in range(100):
        rnd_idx = np.random.choice(idx_list, replace=True, size=data_length)
        test_sample = bin_vphi[rnd_idx]
        sig_vphi = bin_sig_vphi[rnd_idx]
        bootstrapped_means[i] = (test_sample/sig_vphi).sum()/(1/sig_vphi).sum()
    conf_int = np.percentile(bootstrapped_means, [16, 84])

    return (conf_int[1] - conf_int [0])/2