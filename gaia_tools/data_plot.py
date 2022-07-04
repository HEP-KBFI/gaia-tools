from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import mpl_scatter_density
import astropy
import pandas as pd
import corner
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




def display_values(XX, YY, H, mode = None):
    """A function that displays the specific numerical values inside each bin.

    Args:
        XX (array): Bin boundaries in the x-axis.
        YY (array): Bin boundaries in the y-axis.
        H (asarray): The values inside the bins.
        mode (str, optional): The statistic used in the bin. Defaults to None.
    """


    for i in range(YY.shape[0]-1):
        for j in range(XX.shape[1]-1):

            if mode != 'count':
                txt = plt.text((XX[0][j+1] + XX[0][j])/2, (YY.T[0][i+1] + YY.T[0][i])/2, '%.2f' % H.T[i, j],
                     horizontalalignment='center',
                     verticalalignment='center',
                     backgroundcolor='w')

            else:
                txt = plt.text((XX[0][j+1] + XX[0][j])/2, (YY.T[0][i+1] + YY.T[0][i])/2, '%.0f' % H.T[i, j],
                     horizontalalignment='center',
                     verticalalignment='center', backgroundcolor='w')


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
    """A plot to display the bins collapsed along the angular position coordinate phi.

    Args:
        bin_collection (BinCollection obj): Generated BinCollectino object.
        projection_parameter (str): The physical parameter to be observed.
        showBinValues (bool, optional): Shows numerical values in bins. Defaults to True.
        mode (str, optional): The statistic used on the parameter. Defaults to 'mean'.
    """

    parameter = projection_parameter

    XX, YY = bin_collection.bin_boundaries[0:2]

    values = bin_collection.CalculateValues(parameter, mode = mode)

    if values.dtype == 'object':
        values = values.astype('float64')

    fig = plt.figure(figsize = (20,10))
    ax1=plt.subplot(111)
    plot1 = ax1.pcolormesh(XX,YY,values.T)

    if(showBinValues):
        display_values(XX, YY, values, mode)

    # Fix unit displaying later on!
    #cbar = plt.colorbar(plot1,ax=ax1,
                        #pad = .015,
                        #aspect=20,
                        #label='R-Z Bins {0} [{1}]'.format(parameter, 'a.u.'))


    if(mode == 'MLE_std' or mode == 'MLE_mu'):
        projection_parameter = 'v_phi'

    cbar = plt.colorbar(plot1,ax=ax1,
                        pad = .015,
                        aspect = 20,
                        label = '{0} {1}'.format(projection_parameter, mode))

    plt.xticks(XX[0])
    plt.yticks(YY.T[0])


    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #ax1.set_xlabel('$r$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})
    #ax1.set_ylabel('$z$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 18})


    plt.ticklabel_format(axis="x", style="sci", scilimits=(3,3))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(3,3))

    ax1.set_xlabel('r [pc]' , fontdict={'fontsize': 18})
    ax1.set_ylabel('z [pc]', fontdict={'fontsize': 18})


    plt.title("Collapsed Bins in r-z Plane", pad=20, fontdict={'fontsize': 20})

    arrowed_spines(fig, ax1)
    plt.show()


def Velocity_Field_Imshow(bin_collection,
                          title_string = "",
                          arg_notes="",
                          interpolation_type = "gaussian",
                          radii = [0],
                          display_arrows = False,
                          plot_circles=False,
                          save = False):
    """Creates a velocity heatmap from binned data

    Args:
        bin_collection (BinCollection): Binned data returned from the 'bin_data' function in 'data_analysis'
        title_string (str, optional): The title of the plot. Defaults to "".
        arg_notes (str, optional): Subtitle of  the plot. Defaults to "".
        interpolation_type(str, optional): Set the interpolation of the bin values. Defaults to "gaussian".
        radii (list, optional): A list of concentric circle radii which is plotted if 'plot_circles' is set to True. Defaults to [0].
        display_arrows (bool, optional): If True, plots the velocity vector in each bin. Defaults to False.
        plot_circles (bool, optional): If True, plots concentric circles around the Galactic centre. Defaults to False.
        save (bool, optional): If True, saves the figure to disk in '.png' using the 'title_string' as  the file name. Defaults to False.
    """


    H = bin_collection.CalculateValues('v_x')
    H2 = bin_collection.CalculateValues('v_y')

    XX = bin_collection.bin_boundaries[0]
    YY = bin_collection.bin_boundaries[1]




    # Gets the vector coordinates
    VEC_XX, VEC_YY = generate_vector_mesh(bin_collection.bin_boundaries[0], bin_collection.bin_boundaries[1])

    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_xlabel(r'$x$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 15}, labelpad = 25)
    ax.set_ylabel(r'$y$ [{0:latex_inline}]'.format(astropy.units.core.Unit('pc')), fontdict={'fontsize': 15}, labelpad = 25)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    # Gives the hypotenuse of vectors
    M = np.hypot(H.T, H2.T)

    BIN_X_EDGES = bin_collection.bin_boundaries[0][0]
    BIN_Y_EDGES = bin_collection.bin_boundaries[1].T[0]

    dx = (BIN_X_EDGES[0]-BIN_X_EDGES[1])/2.
    dy = (BIN_Y_EDGES[0]-BIN_Y_EDGES[1])/2.
    extent = [BIN_X_EDGES[0], BIN_X_EDGES[-1], BIN_Y_EDGES[0], BIN_Y_EDGES[-1]]

    plt.xticks(XX[0])
    plt.yticks(YY.T[0])


    c = plt.imshow(M, extent = extent, interpolation = interpolation_type, cmap='jet')

    if(display_arrows):
        # The quiver plot with normalised vector lengths
        q = ax.quiver(VEC_XX, VEC_YY, H.T, H2.T, M, cmap=plt.cm.magma_r)


    ax.plot(0, 0, "x", color='red')
    ax.plot(-8178, 0, "*", markersize=20, color='red')

    plt.title(title_string, pad = 45, fontdict={'fontsize': 20})
    plt.suptitle(arg_notes, y=0.93, fontsize=15)

    cbar = plt.colorbar(c, ax=ax, pad = 0.05)
    cbar.set_label(label ='Velocity in bin [km/s]', labelpad= 30, size = 15)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


    # Plot Circles
    if(plot_circles):
        angle = np.linspace( 0 , 2 * np.pi , 150 )

        for i, radius in enumerate(radii):

            radius = radii[i]
            x = radius * np.cos( angle )
            y = radius * np.sin( angle )
            ax.plot( x, y , label='{0} pc'.format(radius))


    plt.legend(loc='upper left')
    plt.grid()

    current_cmap = mpl.cm.get_cmap('jet')
    print(current_cmap)
    current_cmap.set_bad(color='grey')

    if(save):
        plt.savefig(title_string +'.png', dpi=300, format='png')


'''
Generates a velocity vector field from binned data.
Input parameters:
    binned_dict - A dictionary containing all requisite data for displaying the vector field
'''
def generate_velocity_vector_map(bin_collection):


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

    from .data_analysis import transform_to_galcen, get_transformed_data

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

    ax.spines["left"].set_position(("data", xmin))
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
    ax.arrow(xmin, 0., xmax-xmin, 0., fc='black', ec='black', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(xmin, 0, 0., (ymax-ymin)/2, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(xmin, 0, 0., (ymin-ymax)/2, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)



'''
Input - result of MCMCLooper.run_sampler() which is the emcee sampler object
'''
def display_walkers(looper_result,
                    theta_labels = ['r', 'z', 'u', 'v', 'w']):

    # Get data from emcee sampler
    samples_data = looper_result.get_chain()

    num_parameters = len(theta_labels)

    fig, axes = plt.subplots(num_parameters, figsize=(10, 7), sharex=True)

    labels = theta_labels

    for i in range(num_parameters):
        ax = axes[i]
        ax.plot(samples_data[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples_data))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("Step number");

    plt.show()

'''
flat_samples - result from mcmclooper but result is flattened. See mcmclooper class.

theta_labels - list of your parameter names [(string)]
'''
def generate_corner_plot(flat_samples, theta_labels):

    fig = corner.corner(flat_samples, labels=theta_labels);
    plt.show()

    # Fix this in the future, Sven
    return fig


def display_polar_coordinates(phi, r):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    c = ax.scatter(phi, r, cmap='hsv', alpha=0.5)

    plt.title("Polar Coordinates", pad=20, fontdict={'fontsize': 20})
    plt.show()

    return fig



def display_polar_histogram(galcen_data, outpath, n_bins=100, norm_max = 1000, r_limits = (), title = "Polar Plot", is_save=True):
    """A plot which displays a polar histogram of the stars in a galactocentric frame of reference.

    Args:
        galcen_data (Pandas DataFrame): The galactocentric data.
        n_bins (int, optional): Number of bins used in the plot. Defaults to 100.
        norm_max (int, optional): Colormap saturation limit for each bin. Defaults to 1000.
        r_limits (tuple, optional): Minimum and maximum edge of plotted area in r. If empty, it defaults to min, max r in data. Defaults to ().
        title (str, optional): The title string. Defaults to "Polar Plot".

    Returns:
        fig: The returned figure.
    """
    from astropy.visualization.mpl_normalize import ImageNormalize
    from astropy.visualization import LogStretch

    fig= plt.figure(figsize=(10, 10), facecolor='white')

    # Init Data
    phi = galcen_data.phi
    r = galcen_data.r

    if not r_limits:
        min_r = np.min(galcen_data.r)
        max_r = np.max(galcen_data.r)
    else:
        min_r = r_limits[0]
        max_r = r_limits[1]

    plt.ylim(min_r, max_r)

    # Init Bins
    rbins = np.linspace(0, max_r, n_bins)
    abins = np.linspace(-np.pi,np.pi, n_bins)

    norm_hist2d = ImageNormalize(vmin=0., vmax=norm_max, stretch=LogStretch())

    ax = fig.add_subplot(111, projection='polar')
    plt.hist2d(phi, r, bins=(abins, rbins), norm = norm_hist2d)

    plt.title(title, pad=20, fontdict={'fontsize': 20})

    # Set r label background color to black
    plt.setp(ax.get_yticklabels(), backgroundcolor="black")

    # Set r label font color to white
    ax.tick_params(axis="y", colors="white")

    # Configure angle labels
    ax.set_thetamin(360)
    ax.set_thetamax(0)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Number of stars in bin')

    plt.grid()
    #plt.show()

    fig_name = '/sample_distribution_polar_coords'
    if(is_save):
        plt.savefig(outpath + fig_name +'.png', bbox_inches='tight', dpi=300, facecolor='white')



def sample_distribution_galactic_coords(icrs_data, outpath, is_save = True):

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from matplotlib import colors

    c = SkyCoord(ra=list(icrs_data.ra)*u.degree, dec=list(icrs_data.dec)*u.degree, frame='icrs')

    fig = plt.figure(figsize=(16, 8))

    x = c.galactic.l.to_value()
    y = c.galactic.b.to_value()
    h = plt.hist2d(x, y, bins=250, cmin=50, norm=colors.PowerNorm(0.5), zorder=0.5)
    plt.scatter(x, y, alpha=0.05, s=1, color='k', zorder=0)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    plt.colorbar(h[3], pad=0.02, format=fmt, orientation='vertical', label = 'Star density')



    plt.xlabel(r'$l$ [deg]', fontdict={'fontsize' : 16})
    plt.ylabel(r'$b$ [deg]',  fontdict={'fontsize' : 16})

    plt.title("Sample Distribution in Galactic Coordinates\n nstars = {}".format(icrs_data.shape[0]), fontsize=18, pad=15)
    
    fig_name = '/sample_distribution_galactic_coords'
    if(is_save):
        plt.savefig(outpath + fig_name +'.png', bbox_inches='tight', dpi=300, facecolor='white')


def plot_radial_distribution(sample, outpath, is_save=True):

    fig = plt.figure(figsize=(10, 10))

    fig.patch.set_facecolor('white')

    n_bins = 150
    r_min = 0
    r_max = np.max(sample.r_est)

    plt.hist(sample.r_est, bins=np.linspace(r_min, r_max, n_bins))

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    txt="{0} bins defined in the range [{1} - {2}] kpc".format(n_bins, r_min, np.round(r_max))
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.xlabel(r'$r$ (Heliocentric) [pc]', fontdict={'fontsize': 18}, labelpad = 20);
    plt.ylabel('Star count', fontdict={'fontsize': 18}, labelpad = 20);
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()

    plt.rcParams["patch.force_edgecolor"] = True
    plt.rc('font', **{'size':'16'})
    plt.title("Heliocentric Stellar Distances\n nstars = {}".format(sample.shape[0]), pad=20, fontdict={'fontsize': 20})

    fig_name = '/star_density_heliocentric_distribution'
    if(is_save):
        plt.savefig(outpath + fig_name +'.png', bbox_inches='tight', dpi=300, facecolor='white')


def plot_distribution(sample, outpath, parameter, param_min, param_max, cutlines=None, is_save=True):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')

    n_bins = 150
    # param_min = -2000
    # param_max= 2000

    h = plt.hist(sample[parameter], bins=np.linspace(param_min, param_max, n_bins), alpha=1)

    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    txt="{0} bins defined in the range [{1} - {2}] pc".format(n_bins, param_min, param_max)
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    plt.xlabel(r'${}$ [pc]'.format(parameter), fontdict={'fontsize': 18}, labelpad = 20);
    plt.ylabel('Star count', fontdict={'fontsize': 18}, labelpad = 20);
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()

    plt.rcParams["patch.force_edgecolor"] = True
    plt.rc('font', **{'size':'16'})

    plt.title("Star Density Histogram ({})\n nstars = {}".format(parameter, sample.shape[0]), pad=20, fontdict={'fontsize': 20})

    if(cutlines is not None):
        ax.vlines([cutlines[0], cutlines[1]], 0, np.max(h[0]), colors='yellow', linestyles='--')

    fig_name = '/star_density_{}_distribution'.format(parameter)
    if(is_save):
        plt.savefig(outpath + fig_name +'.png', bbox_inches='tight', dpi=300, facecolor='white')
