# This module contains methods for transforming Gaia data

import pandas as pd
import numpy as np
from scipy import stats
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable
from BinCollection import BinCollection
import transformation_constants
import timeit, time

def filter_distance(df, dist, *args, **kwargs):
    """Function for filtering out entries that are further out than some specified distance [pc]
        from Sun's position.

    Args:
        df (DataFrame): Data in ICRS
        dist (float): Max distance

    Returns:
        DataFrame: Filtered DataFrame
    """

    df['distance'] = 1/df.parallax

    # Distance given in pc
    df = df[df.distance <= dist]
    df.reset_index(inplace=True, drop=True)

    return df


'''
OLD ASTROPY FUNCTION

Helper function for 'get_SkyCoord_object' function. Returns QTable containing parallax values.
'''
def get_parallax_table(parallax_series):

    df_parallax = pd.DataFrame(parallax_series, columns="index parallax".split())
    df_parallax.drop('index',axis=1, inplace=True)
    t = QTable.from_pandas(df_parallax)

    return t

'''
Function for instantiating a SkyCoord object with given data.

OLD ASTROPY FUNCTION
'''
def get_SkyCoord_object(df):

    # Gets a distance object to input into SkyCoord object. Need to fix this in the future..
    t = get_parallax_table(df.parallax)
    dist = coord.Distance(parallax=u.Quantity(t['parallax']*u.mas))

    c = coord.SkyCoord(ra=df['ra']*u.degree,
                   dec=df['dec']*u.degree,
                   radial_velocity=df['radial_velocity']*[u.km/u.s],
                   distance=dist,
                   pm_ra_cosdec=df['pmra']*[u.mas/u.yr],
                   pm_dec=df['pmdec']*[u.mas/u.yr],
                   frame = 'icrs')

    return c

# Currently expect data to be in DataFrame format
# TODO: Allow for varying of positional parameters of the Sun
# TODO: Transform back to DataFrame
'''
OLD ASTROPY FUNCTION

Function for transforming data into a galactocentric reference frame using SkyCoord objects and
the 'transform_to' function.

Returns a SkyCoord object
'''
def transform_to_galcen(df, z_sun=17*u.pc, galcen_distance=8.178*u.kpc):

    c = get_SkyCoord_object(df)
    c = c.transform_to(coord.Galactocentric(z_sun=z_sun, galcen_distance=galcen_distance))

    return c

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

    ZZ = (BL_z[0], BL_z[1])

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


def get_collapsed_bins(data, 
                       theta, 
                       BL_r_min, 
                       BL_r_max, 
                       BL_z_min, 
                       BL_z_max, 
                       N_bins = (10, 10), 
                       r_drift = False, 
                       debug=False):
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
        import time, timeit
        tic=timeit.default_timer()
        print("Binning data from galactocentric input data...")
        print("Max r value in DataFrame {0}".format(np.max(data.r)))

    # Fix for newly developed method
    data = data

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
    data['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(data, N_bins, XX, YY, YY, mode='r-z')

    # Generate the bins with respective r-z boundaries
    bin_collection.GenerateBins()

    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for binning data with collapsed bins: {a} sec".format(a=toc-tic))

    return bin_collection


def generate_vector_mesh(XX, YY):
    vec_x = []
    vec_y = []
    vec_z = []

    """Function for finding center points of bins in binned data and then
        creating a meshgrid out of all the point coordinates. The center
        points of bins are the origin points for the vectors.

    Returns:
        tuple: Tuple of x and y arrays
    """
    for i in range(XX.shape[1]-1):
        vec_x.append((XX[0][i+1]+XX[0][i])/2)

    for j in range(YY.shape[0]-1):
        vec_y.append((YY.T[0][j+1]+YY.T[0][j])/2)

    # We create a meshgrid out of all the vector locations
    VEC_XX, VEC_YY = np.meshgrid(vec_x, vec_y)

    return VEC_XX, VEC_YY


def get_transformed_data(data_icrs,
                        include_cylindrical = False,
                        z_0 = transformation_constants.Z_0,
                        r_0 = transformation_constants.R_0,
                        v_sun = transformation_constants.V_SUN,
                        debug = False,
                        is_source_included = False,
                        is_bayes = False):
    """Main function for transforming ICRS data to galactocentric frame.

    Args:
        data_icrs (DataFrame): Gaia data in ICRS
        include_cylindrical (bool, optional): Flag whether to add cylindrical coordinates. Defaults to False.
        z_0 (float, optional): Sun's position over Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's Galactocentric distance. Defaults to transformation_constants.R_0.
        v_sun (tuple, optional): Sun's velocity vector. Defaults to transformation_constants.V_SUN.
        debug (bool, optional): Verbose flag. Defaults to False.
        is_source_included (bool, optional): Flag to carry Gaia's 'source_id' parameter with data. Defaults to False.
        is_bayes (bool, optional): Flag to use Bayesian distance estimates. Defaults to False.

    Returns:
        DataFrame: (Gaia) data in galactocentric frame.
    """

    if(debug):
        tic=timeit.default_timer()
        print("Starting galactocentric transformation loop over all data points.. ")

    # Coordinate vector in galactocentric frame in xyz
    coords =  transform_coordinates_galactocentric(data_icrs, z_0, r_0, is_bayes)

    # Velocity vector in galactocentric frame in xyz
    velocities = transform_velocities_galactocentric(data_icrs, z_0, r_0, v_sun, is_bayes)

    if(include_cylindrical):

        # Using arctan2 which is defined in range [-pi ; pi]
        phi = np.arctan2(coords[:,1],coords[:,0])
        vel_cyl = transform_velocities_cylindrical(velocities, phi)

        cyl_coords = (np.sqrt(coords[:,0]**2 + coords[:,1]**2), phi)

    # Declare DataFrames
    coords_df = pd.DataFrame(np.squeeze(coords, axis=2), columns="x y z".split())
    velocities_df = pd.DataFrame(np.squeeze(velocities, axis=2), columns="v_x v_y v_z".split())

    galcen_df = pd.concat([coords_df, velocities_df], axis=1)

    if(include_cylindrical):
        d = {"r": np.squeeze(cyl_coords[0], axis=1), "phi": np.squeeze(cyl_coords[1], axis=1)}
        coords_df = pd.DataFrame(d)

        # Removing one column because already have v_z
        velocities_df = pd.DataFrame(np.squeeze(vel_cyl[:,0:2], axis=2), columns="v_r v_phi".split())

        df_1 = pd.concat([coords_df, velocities_df], axis=1)
        galcen_df = pd.concat([galcen_df, df_1], axis=1)

    if(is_source_included):

        if not 'source_id' in data_icrs.columns:
            print("Error! Source ID column not found in input DataFrame!")

        galcen_df['source_id'] = data_icrs.source_id


    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for data coordinate transformation: {a} sec".format(a=toc-tic))

    return galcen_df


def transform_coordinates_galactocentric(data_icrs, z_0 = transformation_constants.Z_0, r_0 = transformation_constants.R_0, is_bayes = False):
    """This function uses input ICRS data and outputs data in cartesian (x,y,z) coordinates and in galactocentric frame of reference.

    Args:
        data_icrs (DataFrame): DataFrame of ICRS coordinates
        z_0 (float, optional): Sun's height over Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's Galactocentric distance. Defaults to transformation_constants.R_0.
        is_bayes (bool, optional): Flag for using pre-computed (Bayesian) distance estimates. Defaults to False.

    Returns:
        ndarray: Array of Cartesian coordinates of shape (n,3,1).
    """

    #TODO: Add ASSERT checks on function input parameters.
    # ra dec can only be in a specific range

    # Number of data points
    n = len(data_icrs)

    # Going from DEG -> RAD
    ra = np.deg2rad(data_icrs.ra)
    dec = np.deg2rad(data_icrs.dec)

    if(is_bayes):
        c1 = data_icrs.r_est

    else:
        # from kpc -> pc
        k1 = transformation_constants.k1

        # Declaring constants to reduce process time
        c1 = k1/data_icrs.parallax

    cosdec = np.cos(dec)

    # Initial cartesian coordinate vector in ICRS
    coordxyz_ICRS = np.array([[(c1)*np.cos(ra)*cosdec],
                      [(c1)*np.sin(ra)*cosdec],
                       [(c1)*np.sin(dec)]])

    coordxyz_ICRS = coordxyz_ICRS.T.reshape(n,3,1, order = 'A')

    # Using M1, M2, M3 for transparency in case of bugs
    M1 = transformation_constants.A @ coordxyz_ICRS
    M2 = M1 - np.array([[r_0],
                        [0],
                        [0]])
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2

    # Return is a np.array of shape (n,3,1)
    return M3

'''
This function uses input ICRS data and outputs data in cartesian (v_x,v_y,v_z) velocity vector components and in galactocentric frame of reference.
'''
def transform_velocities_galactocentric(data_icrs, z_0 = transformation_constants.Z_0, r_0 = transformation_constants.R_0, v_sun = transformation_constants.V_SUN, is_bayes = False):
    """Function for transforming proper motions with radial velocities to Cartesian velocity vector components in galactocentric frame.

    Args:
        data_icrs (DataFrame): DataFrame in ICRS
        z_0 (float, optional): Sun's position over Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's Galactocentric distance. Defaults to transformation_constants.R_0.
        v_sun (tuple, optional): Sun's velocity vector. Defaults to transformation_constants.V_SUN.
        is_bayes (bool, optional): Flag for using pre-computed (Bayesian) distance estimates. Defaults to False.

    Returns:
        ndarray: Cartesian velocity components of shape (n,3,1)
    """
    # Number of data points
    n = len(data_icrs)

    # Going from DEG -> RAD
    ra = np.deg2rad(data_icrs.ra)
    dec = np.deg2rad(data_icrs.dec)

    # from 1/yr -> km/s
    k2 = transformation_constants.k2

    if(is_bayes):

        # Assign r estiamtes to c2
        c2 = data_icrs.r_est
        c2 = k2*(data_icrs.r_est/1000)

    else:

        # Declaring constants to reduce process time
        c2 = k2/data_icrs.parallax

    # Initial velocity vector in ICRS in units km/s
    v_ICRS = np.array([[data_icrs.radial_velocity],
                      [(c2)*data_icrs.pmra],
                      [(c2)*data_icrs.pmdec]])

    v_ICRS = v_ICRS.T.reshape(n,3,1, order = 'A')

    B = transformation_constants.get_b_matrix(ra.to_numpy(), dec.to_numpy())
    B = B.reshape(n,3,3, order = 'A')

    # Using M1, M2, M3, .. for transparency in case of bugs
    M1 = B @ v_ICRS
    M2 = transformation_constants.A @ M1
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2

    # Return is a np.array of shape (n,3,1)
    M4 = M3 + v_sun

    return M4

def transform_velocities_cylindrical(velocities_xyz, phi):
    """Transforms Cartesian velocities to cylindrical

    Args:
        velocities_xyz (np.array): Cartesian velocity array
        phi (np.array): Array of phi coordinates

    Returns:
        np.array: Array of cylindrical velocity components.
    """
    v_cylindrical = transformation_constants.get_cylindrical_velocity_matrix(phi) @ velocities_xyz

    return v_cylindrical





