from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

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

'''
A 2d point source density plot on the x-y plane.
Good for visualising large data sets (number of points > 50 000)
Input parameters:
    galcen - SkyCoord object in galactocentric frame
    vmax - maximum number of points normalised per pixel
'''
def point_density(galcen, vmax):
    
    norm = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

    xkoord = -galcen.x.value
    ykoord = galcen.y.value
   
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
    density = ax.scatter_density(xkoord, ykoord, norm=norm, dpi = 200, cmap=plt.cm.jet)
    
    fig.colorbar(density, label='Number of sources per pixel')
    
    plt.title("Point Source Density (Galactocentric)", pad=20, fontdict={'fontsize': 20})
    plt.grid()

    # TODO: Make x and y limits changeable
    ax.set_xlim(-20000, 20000)
    ax.set_ylim(-20000, 20000)

    ax.set_xlabel('$x$ [{0:latex_inline}]'.format(galcen.x.unit))
    ax.set_ylabel('$y$ [{0:latex_inline}]'.format(galcen.y.unit))

    plt.show()
    

def main():
    print("I am the main function")