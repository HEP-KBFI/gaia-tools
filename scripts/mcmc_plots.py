import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import reduce
import emcee
import corner
sys.path.append("../gaia_tools/")



def plot_walkers(sampler_path, 
                burn_in = 0,
                extra_dim_labels = [], 
                plot_name = 'Unnamed', 
                is_save=False):
    
    reader = emcee.backends.HDFBackend(sampler_path, read_only=True)
    samples_data = reader.get_chain(discard = burn_in)
    print("Sampler shape: {}".format(samples_data.shape))

    xdf = [num for num in range(0, samples_data.shape[2], 1)]
    theta_labels = [r'$V_{c%s}$' %str(i+1) for i in xdf]
    
    if(len(extra_dim_labels) != 0):
        theta_labels = theta_labels[:len(theta_labels)-len(extra_dim_labels)]
        for i in range(len(extra_dim_labels)):
            theta_labels.append(extra_dim_labels[i])
    num_parameters = len(theta_labels)

    fig, axes = plt.subplots(num_parameters, figsize=(10, 14), sharex=True)

    for i in range(num_parameters):
        ax = axes[i]
        ax.plot(samples_data[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples_data))
        ax.set_ylabel(theta_labels[i], fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)      
        ax.tick_params(axis='both', which='major', labelsize=16)
        axes[-1].set_xlabel("Step number", labelpad = 20, fontsize=18)
    if(is_save): 
        plt.savefig(plot_name, dpi=300)
    plt.show()

def compare_mcmc_runs(samplers, bin_idx, discard=0):
    reader_1 = emcee.backends.HDFBackend(samplers[0], read_only=True)
    reader_2 = emcee.backends.HDFBackend(samplers[1], read_only=True)

    smps_data_1 = reader_1.get_chain(discard=discard)
    smps_data_2 = reader_2.get_chain(discard=discard)

    print("Sampler 1 shape: {}".format(smps_data_1.shape))
    print("Sampler 2 shape: {}".format(smps_data_2.shape))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10), sharey=True)

    bin_idx = bin_idx

    ax1.plot(smps_data_1[:, :, bin_idx], "k", alpha=0.3)
    ax2.plot(smps_data_2[:, :, bin_idx], "k", alpha=0.3)

    ax1.ticklabel_format(useOffset=False)
    ax2.ticklabel_format(useOffset=False)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylabel('Value', labelpad = 20, fontsize=20)
    ax1.set_xlabel("Step number", labelpad = 20, fontsize=20)
    ax2.set_xlabel("Step number", labelpad = 20, fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_corner(path, burn_in = 0, plot_name = 'Unnamed', is_save=False):
    
    reader = emcee.backends.HDFBackend(path, read_only=True)
    samples_data = reader.get_chain(discard=burn_in)

    xdf = [num for num in range(0, samples_data.shape[2], 1)]
    theta_labels = [r'$V_{c%s}$' %str(i+1) for i in xdf]
    reader = emcee.backends.HDFBackend(path, read_only=True)
    flatchain = reader.get_chain(flat=True, discard = burn_in)

    print(flatchain.shape)

    fig = corner.corner(flatchain, 
                        labels=theta_labels,
                        levels = (1-np.exp(-0.5), 1-np.exp(-2)),
                        plot_datapoints = False,
                        plot_density = False,
                        fill_countours = False,
                        smooth = 0.5,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, 
                        title_kwargs={"fontsize": 11})

    if(is_save): 
        plt.savefig(plot_name, dpi=300)
    plt.show()


