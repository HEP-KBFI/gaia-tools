import transformation_constants
import numpy as np



def transform_coordinates_galactocentric(data_icrs, 
                                        z_0 = transformation_constants.Z_0, 
                                        r_0 = transformation_constants.R_0, 
                                        is_bayes = False,
                                        NUMPY_LIB = np,
                                        dtype = np.float64):

    """This function uses input ICRS data and outputs data in cartesian (x,y,z) coordinates and in galactocentric frame of reference.

    Args:
        data_icrs (DataFrame): DataFrame of ICRS coordinates
        z_0 (float, optional): Sun's height over Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's Galactocentric distance. Defaults to transformation_constants.R_0.
        is_bayes (bool, optional): Flag for using pre-computed (Bayesian) distance estimates. Defaults to False.

    Returns:
        ndarray: Array of Cartesian coordinates of shape (n,3,1).
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
    M2 = NUMPY_LIB.matmul(transformation_constants.get_A_matrix(NUMPY_LIB, dtype), NUMPY_LIB.matmul(B, v_ICRS))
    M3 = NUMPY_LIB.matmul(transformation_constants.get_H_matrix(z_0, r_0, NUMPY_LIB), M2)

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
                        debug = False,
                        is_source_included = False,
                        is_bayes = False,
                        NUMPY_LIB = np,
                        dtype = np.float64):


    # Coordinate and velocity vector in galactocentric frame in xyz
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
    
    if(include_cylindrical):
        # Using arctan2 which is defined in range [-pi ; pi]
        phi = NUMPY_LIB.arctan2(coords[:,1],coords[:,0])
        vel_cyl = transform_velocities_cylindrical(velocities, phi, NUMPY_LIB, dtype)
        coords_cyl = (NUMPY_LIB.sqrt(coords[:,0]**2 + coords[:,1]**2), phi)
        
    galcen_out = NUMPY_LIB.concatenate((NUMPY_LIB.squeeze(coords, axis=2), NUMPY_LIB.squeeze(velocities, axis=2)), axis=1)
    coords_cyl = NUMPY_LIB.squeeze(NUMPY_LIB.asarray(coords_cyl).T, axis=0)
    vel_cyl = NUMPY_LIB.squeeze(vel_cyl, axis=2)[:,0:2]
    galcen_out = NUMPY_LIB.concatenate((galcen_out, coords_cyl, vel_cyl), axis=1)
    
    return galcen_out