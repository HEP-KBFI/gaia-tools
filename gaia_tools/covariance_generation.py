'''
Module for storing functions and constants related to
generating and transforming covariance matrices.
'''

import numpy as np
from . import transformation_constants
import pandas as pd
import timeit

ERROR_NAMES = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']


def generate_covmatrices(df,
                         df_crt = None,
                         transform_to_galcen = False,
                         transform_to_cylindrical = False,
                         z_0 = transformation_constants.Z_0,
                         r_0 = transformation_constants.R_0,
                         is_bayes = False,
                         is_unpack_velocity = False,
                         debug = False):
    """Generate covariance matrices for ICRS data and propagates to galactocentric Cartesian/cylindrical modes if flagged so.

    Args:
        df (DataFrame): The Gaia ICRS data.
        df_crt (DataFrame, optional): The Gaia data in galactocentric Cartesian coordinates. Defaults to None.
        transform_to_galcen (bool, optional): Set true to propagate covariance information to Cartesian coordinates. Defaults to False.
        transform_to_cylindrical (bool, optional): Set true to propagate covariance information to cylindrical coordinates.
                                                    NB! Requires Cartesian covariance matrix. Defaults to False.
        z_0 (float, optional): Sun's height over the Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's distance from the Galactic centre. Defaults to transformation_constants.R_0.
        is_bayes (bool, optional): Set True if using distance estimates and not parallax. Defaults to False.
        debug (bool, optional): Set True for verbose. Defaults to False.

    Returns:
        DataFrame: Returns (transformed) covariance matrices with 'source_id'-s.
    """

    Z_0 = z_0
    R_0 = r_0

    assert len(df) > 0, "Error! No data found in input DataFrame!"
    assert len(df_crt) > 0, "Error! No data found in input galactocentric DataFrame!"

    if(debug):
        print("Generating covariance matrices from input data..")
        tic=timeit.default_timer()

    # Get covariance matrix from ICRS coordinates
    C = generate_covmat(df)

    if(transform_to_galcen is True):

        if(is_bayes == True):
            data_array = df[["ra", "dec","r_est","pmra","pmdec","radial_velocity"]].to_numpy()

        else:
            data_array = df[["ra", "dec","parallax","pmra","pmdec","radial_velocity"]].to_numpy()

        if isinstance(data_array, np.ndarray):
            C = transform_cov_matrix(C, data_array, "Cartesian", Z_0, R_0, is_bayes=is_bayes)
        else:
            print("Data is not a numpy array!")
            return

    if(transform_to_cylindrical is True):

        assert transform_to_galcen, "Must first transform to galactocentric frame!"

        data_array = df_crt[["x", "y","r","phi","v_r","v_phi"]].to_numpy()

        if isinstance(data_array, np.ndarray):
            C = transform_cov_matrix(C, data_array, "Cylindrical", Z_0, R_0)
        else:
            print("Data is not a numpy array!")
            return


    if(is_unpack_velocity):
        covariance_data = {"source_id": df_crt.source_id,
                        "sig_vphi": C[:, 4, 4],
                        "sig_vr": C[:, 3, 3]}
        cov_df = pd.DataFrame(covariance_data)

    else:
        # Unpack covariance matrices to list
        # TODO: Figure out a more efficient way to do this!!
        cov_list = list(C)

        d = {"source_id": df.source_id, "cov_mat": cov_list}
        cov_df = pd.DataFrame(d)

    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for covariance matrix generation and transformation: {a} sec".format(a=toc-tic))

    return cov_df


def generate_galactocentric_covmat(df,
                                is_bayes,
                                Z_0 = transformation_constants.Z_0,
                                R_0 = transformation_constants.R_0):

    """
    Generates a 6x6 covariance matrix in the Galactic frame from a dataframe in the ICRS frame.

    Parameters:
    - df: a Pandas dataframe with columns "ra", "dec", "r_est" (if is_bayes is True) or "parallax",
          "pmra", "pmdec", and "radial_velocity".
    - is_bayes: a boolean indicating whether the df contains distances (True) or measured parallaxes (False).
    - Z_0: optional, the distance of the Sun above the midplane of the galaxy in parsecs. Default is
           the value in the transformation_constants module.
    - R_0: optional, the distance from the Sun to the center of the galaxy in parsecs. Default is
           the value in the transformation_constants module.

    Returns:
    - A NumPy array with shape (n, 6, 6) containing the covariance matrices of stars in the Galactic frame.
    """

    # Get covariance matrix from ICRS coordinates
    C = generate_covmat(df, not is_bayes)

    if(is_bayes == True):
        data_array = df[["ra", "dec","r_est","pmra","pmdec","radial_velocity"]].to_numpy()
    else:
        data_array = df[["ra", "dec","parallax","pmra","pmdec","radial_velocity"]].to_numpy()

    if isinstance(data_array, np.ndarray):
        C = transform_cov_matrix(C, data_array, "Cartesian", Z_0, R_0, is_bayes=is_bayes)
    else:
        print("Data is not a numpy array!")
        return

    return C

def transform_cov_galactocentric(df, 
                                C, 
                                is_bayes, 
                                Z_0 = transformation_constants.Z_0,
                                R_0 = transformation_constants.R_0):

    if(is_bayes == True):
        data_array = df[["ra", "dec","r_est","pmra","pmdec","radial_velocity"]].to_numpy()

    else:
        data_array = df[["ra", "dec","parallax","pmra","pmdec","radial_velocity"]].to_numpy()

    if isinstance(data_array, np.ndarray):
        C = transform_cov_matrix(C, data_array, "Cartesian", Z_0, R_0, is_bayes=is_bayes)
    else:
        print("Data is not a numpy array!")
        return

    return C


def transform_cov_cylindirical(df_crt, 
                                C,
                                Z_0 = transformation_constants.Z_0,
                                R_0 = transformation_constants.R_0):

    data_array = df_crt[["x", "y","r","phi","v_r","v_phi"]].to_numpy()

    C = transform_cov_matrix(C, data_array, "Cylindrical", Z_0, R_0)

    covariance_data = {"source_id": df_crt.source_id,
                        "sig_vphi": C[:, 4, 4],
                        "sig_vr": C[:, 3, 3]}
    cov_df = pd.DataFrame(covariance_data)

    return cov_df


def generate_covmat(df, is_parallax = False):
    """
    Generates a 6x6 covariance matrix in the celestial frame from a dataframe of Gaia data.

    Parameters:
    - df: a Pandas dataframe with columns "ra", "dec", "parallax", "pmra", "pmdec", and "radial_velocity".
           The dataframe should also have columns ending in "_error" for each of the above parameters,
           and columns ending in "_corr" for pairs of parameters that are correlated.

    Returns:
    - A NumPy array with shape (n, 6, 6) where n is the number of rows in df, containing the covariance
      matrices in the celestial frame.
    """

    n = len(df)

    # Declare empty matrix array
    C = np.zeros((n, 6, 6))

    # Possible parameter names
    if(is_parallax):
        names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    else:
        names = ['ra', 'dec', 'r_est', 'pmra', 'pmdec']

    # For Diagonal Elements
    for i, name in enumerate(names):
            ext = names[i] + "_error"

            # TODO: Move this outside of this function!
            if not ext in df.columns:
                print("{0} not in data!".format(ext))
                return

            err = getattr(df, ext)

            if(name == 'ra' or name == 'dec'):

                # This converts from [mas] to [deg]
                err = err/(3.6 * 10**6)

            C[:, i, i] = err ** 2

    # For Radial Velocity Element
    C[:,5,5] = getattr(df,'radial_velocity_error') ** 2

    # For Rest of the Elements
    for i, name1 in enumerate(names):
                for j, name2 in enumerate(names):

                    if j <= i:
                        continue

                    if((is_parallax==False) & (name1 == names[2] or name2 == names[2])):
                        continue

                    ext = "{0}_{1}_corr".format(name1, name2)

                    # TODO: Move this outside of this function!
                    if not ext in df.columns:
                        print("{0} not in data!".format(ext))
                        return

                    corr = getattr(df,ext)

                    # Sqrt because it accesses values from the main diagonal which are squared.
                    C[:, i, j] = corr * np.sqrt(C[:,i, i] * C[:,j, j])
                    C[:, j, i] = C[:,i, j]

    return C


def transform_cov_matrix(C, 
                        df, 
                        coordinate_system, 
                        z_0 = transformation_constants.Z_0, 
                        r_0 = transformation_constants.R_0, 
                        is_bayes = False,
                        NUMPY_LIB = np, 
                        dtype = np.float64):

    """Transforms an array of covariance matrices to specified coordinate system.

    Args:
        C (Array): Arrays of covariance matrices
        df (DataFrame): DataFrame of Gaia data used for generating Jacobian matrices for each observation.
        coordinate_system (str): Specified coordinate system, either "Cartesian" or "Cylindrical".
        z_0 (float, optional): Sun's height over Galactic plane. Defaults to transformation_constants.Z_0.
        r_0 (float, optional): Sun's distance from Galactic centre. Defaults to transformation_constants.R_0.
        is_bayes (bool, optional): Set True if using distance estimates instead of parallaxes. Defaults to False.

    Returns:
        Array: Returns array of transformed covariance matrices.
    """

    if(is_bayes == True):
        # Grabs the correct Jacobian for every point in data set. Of shape (n, 6, 6).
        J = transformation_constants.get_jacobian_bayes(df, 
                                                        coordinate_system, 
                                                        Z_0 = z_0, 
                                                        R_0 = r_0,
                                                        NUMPY_LIB = NUMPY_LIB, 
                                                        dtype = dtype)
    else:
        # Grabs the correct Jacobian for every point in data set. Of shape (n, 6, 6).
        J = transformation_constants.get_jacobian(df, 
                                                coordinate_system, 
                                                Z_0 = z_0, 
                                                R_0 = r_0,
                                                NUMPY_LIB = NUMPY_LIB, 
                                                dtype = dtype)

    J = J.T.reshape(len(df), 6, 6, order = 'A').swapaxes(1,2)
    J_trunc= J.reshape(len(df),6,6, order = 'A').swapaxes(1,2)

    C_transformed = NUMPY_LIB.matmul(NUMPY_LIB.matmul(J, C), J_trunc)

    return C_transformed
