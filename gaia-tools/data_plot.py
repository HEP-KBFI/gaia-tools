from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import mpl_scatter_density
import astropy
from data_analysis import generate_vector_mesh
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from BinCollection import BinCollection

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
   
'''
A function that displays the specific numerical values inside each bin.
'''
def display_values(XX, YY, H):
    for i in range(YY.shape[0]-1):
        for j in range(XX.shape[0]-1):
            plt.text((XX[0][j+1] + XX[0][j])/2, (YY.T[0][i+1] + YY.T[0][i])/2, '%.2f' % H.T[i, j],
                 horizontalalignment='center',
                 verticalalignment='center')

'''
A plot which enables the user to see the bins created by the 'bin_data' functions in the 
data analysis module. It takes in the histogram data and does a colormesh with colours 
mapped to the value inside the 2D bin.
'''
def display_mean_velocity(bin_collection, projection_parameter, showBinValues = True):

    parameter = projection_parameter

    XX, YY = bin_collection.bin_boundaries

    values = bin_collection.CalculateValues(parameter)

    fig = plt.figure(figsize = (10,10))
    ax1=plt.subplot(111)
    plot1 = ax1.pcolormesh(XX, YY, values.T)

    if(showBinValues):
        display_values(XX, YY, values)

    cbar = plt.colorbar(plot1,ax=ax1, 
                        pad = .015, 
                        aspect=10, 
                        label='2D Bins Velocity V{0}[{1}]'.format(parameter, 'km/s'))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax1.set_xlabel('$x$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    ax1.set_ylabel('$y$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})

    plt.title("2D Bins Velocity V{0}".format(parameter), pad=20, fontdict={'fontsize': 20})
    
    plt.show()


'''
Generates a velocity field from binned data.
Input parameters:
    binned_dict - A dictionary containing all requisite data for displaying the vector field
'''
def generate_velocity_map(bin_collection):
 

    H = bin_collection.CalculateValues('v_x')
    H2 = bin_collection.CalculateValues('v_y')

    # Gets the vector coordinates
    VEC_XX, VEC_YY = generate_vector_mesh(bin_collection.bin_boundaries[0], bin_collection.bin_boundaries[1])

    fig, ax = plt.subplots(figsize = (7,7))

    # Gives the hypotenuse of vectors
    M = np.hypot(H.T, H2.T)

    norm = mpl.colors.Normalize()
    norm.autoscale(M)
    cm = mpl.cm.jet

    sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])

    # The quiver plot with normalised vector lengths
    q = ax.quiver(VEC_XX, VEC_YY, H.T, H2.T, M, cmap=plt.cm.jet)

    plt.colorbar(sm)

    # Formats x-y axis in scientific notation
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # TODO: Implement for arbitrary limits
    plt.xlim(-20000,20000)
    plt.ylim(-20000,20000)

    ax.set_xlabel('$x$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    ax.set_ylabel('$y$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    
    plt.grid()
    plt.show()

