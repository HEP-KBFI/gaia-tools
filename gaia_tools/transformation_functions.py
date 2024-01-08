import transformation_constants
import numpy as np
import pandas as pd

"""These functions transform coordinates and velocities from the International Celestial Reference System (ICRS) to a galactocentric frame of reference. The ICRS is a reference frame that is centered at the solar system barycenter and is aligned with the celestial sphere. The galactocentric frame of reference is centered on the Galactic Center and has the Galactic plane as its equatorial plane.

The transform_coordinates_galactocentric function takes as input a DataFrame of ICRS coordinates and returns an array of Cartesian coordinates in the galactocentric frame of reference. The input DataFrame should have columns for the right ascension (RA), declination (dec), and distance of each object from the solar system barycenter. The function also takes optional arguments for the height of the Sun above the Galactic plane (z_0) and the Galactocentric distance of the Sun (r_0). If is_bayes is set to True, the function will use pre-computed distance estimates for the objects. Otherwise, it will compute the distance using the parallaxes provided in the input DataFrame.

The transform_velocities_galactocentric function takes as input a DataFrame of ICRS coordinates and velocities, and returns an array of Cartesian velocity vector components in the galactocentric frame of reference. The input DataFrame should have columns for the right ascension, declination, distance, proper motion in RA, and proper motion in dec of each object, as well as the radial velocity. The function also takes optional arguments for the height of the Sun above the Galactic plane (z_0), the Galactocentric distance of the Sun (r_0), and the velocity vector of the Sun (v_sun). If is_bayes is set to True, the function will use pre-computed distance estimates for the objects. Otherwise, it will compute the distance using the parallaxes provided in the input DataFrame.

Both functions make use of the transformation_constants module and the NumPy library (imported as np) to perform the coordinate and velocity transformations. They also have a dtype argument that allows the user to specify the data type of the output array. By default, the output is a NumPy float64 array.
"""

INPUT_COLUMNS_REST = ['source_id', 'ra', 'dec', 'r_est', 'pmra', 'pmdec', 'radial_velocity']
INPUT_COLUMNS_PRLX = ['source_id', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']

def transform_coordinates_galactocentric(data_icrs, 
                                        z_0 = transformation_constants.Z_0, 
                                        r_0 = transformation_constants.R_0, 
                                        is_bayes = False,
                                        NUMPY_LIB = np,
                                        dtype = np.float64):

    """
    Transforms coordinates from the International Celestial Reference System (ICRS) to a galactocentric frame of reference.

    Parameters
    ----------
    data_icrs : DataFrame
        DataFrame of ICRS coordinates, with columns for right ascension (RA), declination (dec), and distance from the solar system barycenter.
    z_0 : float, optional
        Height of the Sun above the Galactic plane. Default is the value specified in transformation_constants.Z_0.
    r_0 : float, optional
        Galactocentric distance of the Sun. Default is the value specified in transformation_constants.R_0.
    is_bayes : bool, optional
        Flag for using pre-computed (Bayesian) distance estimates. Default is False.
    NUMPY_LIB : numpy module, optional
        NumPy library to be used. Default is the imported NumPy module.
    dtype : numpy data type, optional
        Data type of the output array. Default is np.float64.

    Returns
    -------
    ndarray
        Array of Cartesian coordinates in the galactocentric frame of reference, with shape (n, 3, 1).
    """


    # Number of data points
    n = len(data_icrs)

    # Going from DEG -> RAD
    ra = NUMPY_LIB.deg2rad(data_icrs[:,1])
    dec = NUMPY_LIB.deg2rad(data_icrs[:,2])

    if(is_bayes):
        c1 = data_icrs[:,3]

    else:
        # from kpc -> pc
        k1 = transformation_constants.k1

        # Declaring constants to reduce process time
        c1 = k1/data_icrs[:,3]

    
    cosdec = NUMPY_LIB.cos(dec)

    # Initial cartesian coordinate vector in ICRS
    coordxyz_ICRS = NUMPY_LIB.asarray([[(c1)*NUMPY_LIB.cos(ra)*cosdec],
                      [(c1)*NUMPY_LIB.sin(ra)*cosdec],
                       [(c1)*NUMPY_LIB.sin(dec)]])

    coordxyz_ICRS = coordxyz_ICRS.T.reshape(n,3,1, order = 'A')

    # Using M1, M2, M3 for transparency in case of bugs
    M1 = NUMPY_LIB.matmul(transformation_constants.get_A_matrix(NUMPY_LIB, dtype), coordxyz_ICRS)
    M2 = M1 - NUMPY_LIB.asarray([[r_0],
                        [0],
                        [0]], dtype=NUMPY_LIB.float32)

    M3 = NUMPY_LIB.matmul(transformation_constants.get_H_matrix(z_0, r_0, NUMPY_LIB), M2)
   

    # Return is a np.array of shape (n,3,1)
    return M3

'''
This function uses input ICRS data and outputs data in cartesian (v_x,v_y,v_z) velocity vector components and in galactocentric frame of reference.
'''
def transform_velocities_galactocentric(data_icrs, 
                                z_0 = transformation_constants.Z_0, 
                                r_0 = transformation_constants.R_0, 
                                v_sun = transformation_constants.V_SUN, 
                                is_bayes = False,
                                NUMPY_LIB = np,
                                dtype = np.float64):
    """
    Transforms proper motions with radial velocities to Cartesian velocity vector components in a galactocentric frame of reference.

    Parameters
    ----------
    data_icrs : DataFrame
        DataFrame of ICRS coordinates and velocities, with columns for right ascension (RA), declination (dec), distance from the solar system barycenter, proper motion in RA, proper motion in dec, and radial velocity.
    z_0 : float, optional
        Height of the Sun above the Galactic plane. Default is the value specified in transformation_constants.Z_0.
    r_0 : float, optional
        Galactocentric distance of the Sun. Default is the value specified in transformation_constants.R_0.
    v_sun : tuple, optional
        Velocity vector of the Sun. Default is the value specified in transformation_constants.V_SUN.
    is_bayes : bool, optional
        Flag for using pre-computed (Bayesian) distance estimates. Default is False.
    NUMPY_LIB : numpy module, optional
        NumPy library to be used. Default is the imported NumPy module.
    dtype : numpy data type, optional
        Data type of the output array. Default is np.float64.

    Returns
    -------
    ndarray
        Array of Cartesian velocities in the galactocentric frame of reference, with shape (n, 3, 1).
    """

    # Number of data points
    n = len(data_icrs)

    # Going from DEG -> RAD
    ra = NUMPY_LIB.deg2rad(data_icrs[:,1])
    dec = NUMPY_LIB.deg2rad(data_icrs[:,2])

    # from 1/yr -> km/s
    k2 = transformation_constants.k2

    if(is_bayes):
        # Assign r estimates to c2
        c2 = k2*(data_icrs[:,3]/1000)

    else:
        # Declaring constants to reduce process time
        c2 = k2/data_icrs[:,3]

    # Initial velocity vector in ICRS in units km/s
    v_ICRS = NUMPY_LIB.asarray([[data_icrs[:,6]],
                      [(c2)*data_icrs[:,4]],
                      [(c2)*data_icrs[:,5]]])

    v_ICRS = v_ICRS.T.reshape(n,3,1, order = 'A')
    if(NUMPY_LIB == np):
        B = transformation_constants.get_b_matrix(ra, dec)
    else:
        B = transformation_constants.get_b_matrix(ra, dec, NUMPY_LIB, dtype)
    B = B.reshape(n,3,3, order = 'A')

    # Using M1, M2, M3, .. for transparency in case of bugs
    M2 = NUMPY_LIB.matmul(transformation_constants.get_A_matrix(NUMPY_LIB, dtype), NUMPY_LIB.matmul(B, v_ICRS), dtype=dtype)
    M3 = NUMPY_LIB.matmul(transformation_constants.get_H_matrix(z_0, r_0, NUMPY_LIB), M2, dtype=dtype)

    # Return is a np.array of shape (n,3,1)
    M4 = M3 + NUMPY_LIB.asarray(v_sun, dtype=dtype)

    return M4

def transform_velocities_cylindrical(velocities_xyz, phi, NUMPY_LIB, dtype):
    """Transforms Cartesian velocities to cylindrical

    Args:
        velocities_xyz (np.array): Cartesian velocity array
        phi (np.array): Array of phi coordinates

    Returns:
        np.array: Array of cylindrical velocity components.
    """
    v_cylindrical = NUMPY_LIB.matmul(transformation_constants.get_cylindrical_velocity_matrix(phi, NUMPY_LIB, dtype), velocities_xyz)

    return v_cylindrical


def get_transformed_data(data_icrs,
                        include_cylindrical = False,
                        z_0 = transformation_constants.Z_0,
                        r_0 = transformation_constants.R_0,
                        v_sun = transformation_constants.V_SUN,
                        is_bayes = False,
                        is_output_frame = False,
                        is_source_included = False,
                        NUMPY_LIB = np,
                        dtype = np.float64):

    """
    Transforms a set of data in ICRS coordinates to a galactocentric frame of reference, and optionally calculates
    cylindrical coordinates and velocities.
    
    Parameters:
    - data_icrs: An array of ICRS coordinates and velocities.
    - include_cylindrical: A boolean flag indicating whether cylindrical coordinates and velocities should be included in the output.
    - z_0: A constant used in the transformation.
    - r_0: A constant used in the transformation.
    - v_sun: A constant used in the transformation.
    - is_bayes: A boolean flag indicating whether Bayesian priors should be used in the transformation.
    - NUMPY_LIB: A reference to the NumPy library, used for numerical computations.
    - dtype: The data type to use for the calculations.
    
    Returns:
    An array of galactocentric coordinates and velocities, with optional cylindrical coordinates and velocities included.
    """

    if(isinstance(data_icrs, pd.DataFrame)):
        # Filter out unneeded columns
        if(is_bayes):
            data_icrs = (data_icrs[INPUT_COLUMNS_REST]).to_numpy(dtype=dtype)
        else:
            data_icrs = (data_icrs[INPUT_COLUMNS_PRLX]).to_numpy(dtype=dtype)

    # Coordinate and velocity vectors in galactocentric frame in xyz
    coords =  transform_coordinates_galactocentric(data_icrs, 
                                                        z_0, 
                                                        r_0, 
                                                        is_bayes, 
                                                        NUMPY_LIB,
                                                        dtype)

    velocities = transform_velocities_galactocentric(data_icrs, 
                                                    z_0, 
                                                    r_0, 
                                                    v_sun, 
                                                    is_bayes,
                                                    NUMPY_LIB,
                                                    dtype)

    galcen_out = NUMPY_LIB.concatenate((NUMPY_LIB.squeeze(coords, axis=2), NUMPY_LIB.squeeze(velocities, axis=2)), axis=1)

    # Add cylindrical parameters if flagged
    if(include_cylindrical):
        # Using arctan2 which is defined in range [-pi ; pi]
        phi = NUMPY_LIB.arctan2(coords[:,1],coords[:,0])
        vel_cyl = transform_velocities_cylindrical(velocities, phi, NUMPY_LIB, dtype)
        coords_cyl = (NUMPY_LIB.sqrt(coords[:,0]**2 + coords[:,1]**2), phi)
        
        vel_cyl = NUMPY_LIB.squeeze(vel_cyl, axis=2)[:,0:2]
        coords_cyl = NUMPY_LIB.squeeze(NUMPY_LIB.asarray(coords_cyl).T, axis=0)
        galcen_out = NUMPY_LIB.concatenate((galcen_out, coords_cyl, vel_cyl), axis=1)

    if(is_source_included):
        print(dtype(data_icrs[:,0]))
        assert dtype == np.float64, "Data type should be float64"
        galcen_out = NUMPY_LIB.concatenate(([galcen_out, NUMPY_LIB.array([data_icrs[:,0]]).T]), axis=1)
    
    # Declare Pandas frame if flagged
    if(is_output_frame):
        galcen_out = build_outframe(galcen_out, include_cylindrical, is_source_included)

    return galcen_out

def build_outframe(galcen_out, include_cylindrical, is_source_included):

    """
    Transform the input array into a Pandas dataframe with the appropriate columns.

    Parameters:
    - galcen_out (ndarray or cupy.ndarray): Array containing the data to be transformed into a dataframe.
    - include_cylindrical (bool): If True, include cylindrical coordinates (r, phi, v_r, v_phi) in the dataframe.
    - is_source_included (bool): If True, include a 'source_id' column in the dataframe.

    Returns:
    - Pandas dataframe with the appropriate columns and data.
    """

    columns = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'r', 'phi', 'v_r', 'v_phi']

    if(include_cylindrical != True):
        columns = columns[0:6]

    if(is_source_included):
        columns.append('source_id')

    if(isinstance(galcen_out, np.ndarray)):
        galcen_out = pd.DataFrame(galcen_out, columns=columns)
    else:
        # Assumes that it is a CuPy array
        # Note that this also migrates the data to CPU
        galcen_out = pd.DataFrame(galcen_out.get(), columns=columns)
    return galcen_out
