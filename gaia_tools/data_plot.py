from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import mpl_scatter_density
import astropy
import pandas as pd
from data_analysis import generate_vector_mesh
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import BinCollection

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

Input parameters
----------------
    galcen - SkyCoord object in galactocentric frame
    vmax - maximum number of points normalised per pixel
'''
# CURRENTLY BROKEN
def point_density(galcen, vmax):
    
    norm = ImageNormalize(vmin=0., vmax=10, stretch=LogStretch())

    x_coord = -galcen.x.value
    y_coord = galcen.y.value
   
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    
    density = ax.scatter_density(x_coord, y_coord, norm=norm, dpi = 200, cmap=plt.cm.jet)
    
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
A 2D histogram to depict point source density in different regions using 2D bins.
Using this to replace the currently broken 'point_density' plot.

'''
def point_density_histogram(galcen, vmax, bin_start = -16000, bin_end = 16000, n_bins = 200):
    
    # Check if data is in DataFrame or Astropy SkyCoords object
    if isinstance(galcen, pd.DataFrame):
        x_coord = [-x for x in galcen.x]
        y_coord = [y for y in galcen.y]
    else:
        x_coord = [-x for x in galcen.x.value]
        y_coord = [y for y in galcen.y.value]

    norm_hist2d = ImageNormalize(vmin=0., vmax=vmax, stretch=LogStretch())
    
    fig = plt.figure(figsize=(10, 10))
    
    plt.hist2d(x_coord, y_coord, bins=np.linspace(bin_start, bin_end, n_bins), norm = norm_hist2d)
    
    plt.xlabel('x [pc]', fontsize=15)
    plt.ylabel('y [pc]', fontsize=15)
    plt.title("2D Histograms of Data Sources", pad=20, fontdict={'fontsize': 20})
    plt.xlim(bin_start, bin_end)
    plt.ylim(bin_start, bin_end)
    plt.grid()

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of stars in bin')

    plt.show()



'''
A function that displays the specific numerical values inside each bin.
'''
def display_values(XX, YY, H):
    for i in range(YY.shape[0]-1):
        for j in range(XX.shape[1]-1):
            plt.text((XX[0][j+1] + XX[0][j])/2, (YY.T[0][i+1] + YY.T[0][i])/2, '%.2f' % H.T[i, j],
                 horizontalalignment='center',
                 verticalalignment='center')

'''
A plot which enables the user to see the bins created by the 'bin_data' functions in the 
data analysis module. It takes in the histogram data and does a colormesh with colours 
mapped to the value inside the 2D bin.
'''
def display_bins(bin_collection, projection_parameter, mode = 'mean', showBinValues = True):

    parameter = projection_parameter

    XX, YY = bin_collection.bin_boundaries[0:2]

    values = bin_collection.CalculateValues(parameter, mode)

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



def plot_collapsed_bins(bin_collection, projection_parameter, showBinValues = True, mode = 'mean'):

    parameter = projection_parameter

    XX, YY = bin_collection.bin_boundaries[0:2]

    values = bin_collection.CalculateValues(parameter, mode = mode)


    fig = plt.figure(figsize = (10,10))
    ax1=plt.subplot(111)
    plot1 = ax1.pcolormesh(XX,YY,values.T)
    
    if(showBinValues):
        display_values(XX, YY, values)

    # Fix unit displaying later on!
    cbar = plt.colorbar(plot1,ax=ax1, 
                        pad = .015, 
                        aspect=20, 
                        label='R-Z Bins {0} [{1}]'.format(parameter, 'a.u.'))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax1.set_xlabel('$r$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    ax1.set_ylabel('$z$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    
    plt.title("Collapsed Bins in r-z Plane", pad=20, fontdict={'fontsize': 20})

    arrowed_spines(fig, ax1)
    plt.show()
   

'''
Generates a velocity vector field from binned data.
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


'''
df - Imported data from CSV
'''
def run_parameter_tests(df, parameter_list):

    from data_analysis import transform_to_galcen, get_transformed_data

    # Generating Transformation With Astropy
    galcen_astropy = transform_to_galcen(df)
    
    # Using our method
    galcen_my = get_transformed_data(df, include_cylindrical = True)


    for parameter in parameter_list:
        parameter_test_plot(galcen_astropy, galcen_my, parameter)



def parameter_test_plot(galcen_astropy, galcen_my, test_parameter):
    
    # Check if data is in DataFrame or Astropy SkyCoords object
    if isinstance(galcen_astropy, pd.DataFrame):
        x_coord = galcen_my[test_parameter]
        y_coord = galcen_astropy[test_parameter]
    else:
        x_coord = galcen_my[test_parameter]
        if(test_parameter == 'x'):
            y_coord = galcen_astropy.x.value

        elif(test_parameter == 'y'):
            y_coord = galcen_astropy.y.value

        elif(test_parameter == 'z'):
            y_coord = galcen_astropy.z.value

        elif(test_parameter == 'v_x'):
            y_coord = galcen_astropy.v_x.value

        elif(test_parameter == 'v_y'):
            y_coord = galcen_astropy.v_y.value

        elif(test_parameter == 'v_z'):
            y_coord = galcen_astropy.v_z.value

    
    # Right-hand transformation
    if(test_parameter == 'x' or test_parameter == 'v_x'):
        x_coord = [-x for x in x_coord]
        y_coord = [-y for y in y_coord]

    # Converstion to lists
    x_coord = [x for x in x_coord]
    y_coord = [y for y in y_coord]

    plot_label = "Testing parameter: {0}".format(test_parameter)

    plt.scatter(x_coord, y_coord, label=plot_label)
    plt.xlabel("Our values [{0}]".format(test_parameter))
    plt.ylabel("Astropy values [{0}]".format(test_parameter))
    plt.legend(loc='upper left')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.grid()
    plt.title("Our transformation VS Astropy", pad=20, fontdict={'fontsize': 20})
    plt.show()

# Displays arrows on plot
def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    ax.spines["left"].set_position(("data", 0.))
    ax.spines["right"].set_position(("data", xmax))
    
    # removing the default axis on all sides:
    for side in ['bottom','top']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./30.*(ymax-ymin) 
    hl = 1./30.*(xmax-xmin)
    lw = 1 # axis line width
    ohg = 0 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(0., 0., xmax-xmin, 0., fc='black', ec='black', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(0, 0, 0., (ymax-ymin)/2, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)
    
    ax.arrow(0, 0, 0., (ymin-ymax)/2, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)



