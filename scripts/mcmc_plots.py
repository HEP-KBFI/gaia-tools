import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import reduce
import emcee
import corner
sys.path.append("../gaia_tools/")
import data_analysis
import transformation_constants
import covariance_generation
from import_functions import import_data

def plot_walkers(sampler_path, burn_in = 0, plot_name = 'Unnamed', is_save=False):
    
    reader = emcee.backends.HDFBackend(sampler_path, read_only=True)
    samples_data = reader.get_chain(discard = burn_in)
    print("Sampler shape: {}".format(samples_data.shape))

    xdf = [num for num in range(0, samples_data.shape[2], 1)]
    theta_labels = [r'$V_{c%s}$' %str(i+1) for i in xdf]

    num_parameters = len(theta_labels)

    fig, axes = plt.subplots(num_parameters, figsize=(10, 7), sharex=True)

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

def plot_corner(reader, theta_labels, burn_in = 0, plot_name = 'Unnamed', is_save=False):
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