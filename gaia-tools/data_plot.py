from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np


def test_func():
    print("I am from data_plot!")

# TODO: Add additional projections: along y- and z-axis
# TODO: Add options for DataFrame format
'''
A simple histogram displaying the distribution of entries along the galactic plane.
Input parameters:
    galcen - SkyCoord object in galactocentric frame
'''
def distribution_hist(galcen):
    
    fig = plt.figure(figsize=(10, 10))

    plt.hist(-galcen.x.value, bins=np.linspace(-10000, 10000, 32))
    plt.xlabel('$x$ [{0:latex_inline}]'.format(galcen.z.unit), fontdict={'fontsize': 18});
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    plt.ylabel('Count', fontdict={'fontsize': 18});
    plt.grid()
    plt.rcParams["patch.force_edgecolor"] = True
    plt.title("Distribution by Distance", pad=20, fontdict={'fontsize': 20})

    plt.show()


def main():
    print("I am the main function")