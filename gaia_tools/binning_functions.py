from scipy import stats
import numpy as np
from BinCollection import BinCollection
import timeit


def bin_data(galcen_data,
            show_bins = False,
            BL_x = (-10000, 10000),
            BL_y = (-10000, 10000),
            BL_z = (-10000, 10000),
            N_bins = (10, 10),
            debug = False):

    """Function for binning data in xyz coordinates. Useful for plotting velocity fields.

    Returns:
        BinCollection: A collection of Bin type objects.
    """

    if(debug):
        import time, timeit
        tic=timeit.default_timer()
        print("Binning data from galactocentric input data...")

    # Define spatial limits.
    galcen_data = galcen_data[(galcen_data.x >= BL_x[0]) & (galcen_data.x <= BL_x[1])]
    galcen_data = galcen_data[(galcen_data.y >= BL_y[0]) & (galcen_data.y <= BL_y[1])]
    galcen_data = galcen_data[(galcen_data.z >= BL_z[0]) & (galcen_data.z <= BL_z[1])]

    # x, y coordinates of the points
    x = galcen_data.x
    y = galcen_data.y

    # Velocity projections of points
    z = galcen_data.v_x
    z2 = galcen_data.v_y

    # Calling the actual binning function
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(x,
                                                            y,
                                                            values = z,
                                                            bins = N_bins,
                                                            range=[[BL_x[0], BL_x[1]], [BL_y[0], BL_y[1]]],
                                                            statistic='mean')

    # Create a meshgrid from the vertices
    XX, YY = np.meshgrid(xedges, yedges)

    ZZ = (np.min(galcen_data.z), np.max(galcen_data.z))

    # Assign a binnumber for each data entry
    galcen_data['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(galcen_data, N_bins, XX, YY, ZZ)

    # Generate the bins with respective x-y boundaries
    bin_collection.GenerateBins()

    # TODO: Generalise this!
    if(show_bins == True):
        from .data_plot import display_bins

        display_bins(bin_collection, 'v_x')
        display_bins(bin_collection, 'v_y')

    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for binning data: {a} sec".format(a=toc-tic))

    return bin_collection


def get_collapsed_bins(data, theta, BL_r_min, BL_r_max, BL_z_min, BL_z_max, N_bins = (10, 10), r_drift = False, debug=False):
    """Returns bin in r - z

    Args:
        data (DataFrame): Data in galactocentric coordinates
        theta (tuple): Tuple of R0 and z0. Used to mitigate data drift.
        BL_r_min (float): Min r edge
        BL_r_max (float): Max r edge
        BL_z_min (float): Min z edge
        BL_z_max (float): Max z edge
        N_bins (tuple, optional): Number of bins in the r-z plane. Defaults to (10, 10).
        r_drift (bool, optional): Try to prevent drift of data along r-z axes. Defaults to False.
        debug (bool, optional): Verbose option. Defaults to False.

    Returns:
        BinCollection: Collection of Bin objects.
    """

    # This assertion doesnt make sense, fix it later
    assert len(data.shape) > 0, "No data!"

    if not 'r' or 'phi' in data.index:
        print("No cylindrical coordinates found!")
        return

    if(debug):
        tic=timeit.default_timer()
        print("Binning data from galactocentric input data...")
        print("Max r value in DataFrame {0}".format(np.max(data.r)))

    # Fix for newly developed method
    plottable_df = data

    # Setup adimensional binning
    if(r_drift):

        data['r_orig'] = data.r
        data['r'] = data.r/theta

        r = data.r
        z = data.z
        
        if(debug):
            print("Points drifted in r + direction {0}".format(len(data[data.r/theta > BL_r_max])))
            print("Points drifted in r - direction {0}".format(len(data[data.r/theta < BL_r_min])))

    else:
        # r and z parameters of points loaded into Series
        r = data.r
        z = data.z

    # Velocity projections of points: NOT NEEDED
    c = data.v_phi

    # Calling the actual binning function
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(r, 
                                                            z, 
                                                            values = c, 
                                                            range = [[BL_r_min, BL_r_max], [BL_z_min, BL_z_max]], 
                                                            bins=N_bins, 
                                                            statistic='mean')

    # Create a meshgrid from the vertices: X, Y -> R, Z
    XX, YY = np.meshgrid(xedges, yedges)

    # Assign a binnumber for each data entry
    plottable_df['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(plottable_df, N_bins, XX, YY, YY, mode='r-z')

    # Generate the bins with respective r-z boundaries
    bin_collection.GenerateBins()

    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for binning data with collapsed bins: {a} sec".format(a=toc-tic))

    return bin_collection


def generate_vector_mesh(XX, YY):

    """Function for finding center points of bins in binned data and then
        creating a meshgrid out of all the point coordinates. The center
        points of bins are the origin points for the vectors.

    Returns:
        tuple: Tuple of x and y arrays
    """
    
    vec_x = []
    vec_y = []

    for i in range(XX.shape[1]-1):
        vec_x.append((XX[0][i+1]+XX[0][i])/2)

    for j in range(YY.shape[0]-1):
        vec_y.append((YY.T[0][j+1]+YY.T[0][j])/2)

    # We create a meshgrid out of all the vector locations
    VEC_XX, VEC_YY = np.meshgrid(vec_x, vec_y)

    return VEC_XX, VEC_YY