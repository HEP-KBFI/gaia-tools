'''
File for storing functions and constants related to generating and transforming
data covariance matrices.
'''

import numpy as np
import transformation_constants
import pandas as pd
import time, timeit

ERROR_NAMES = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']

'''
Main function for generating covariance matrices and transforming them.


Function that iterates over the DataFrame and appends covariances matrices to a 
dictonary with 'the source_id' as key.
'''
def generate_covmatrices(df, 
                         df_crt = None, 
                         transform_to_galcen = False, 
                         transform_to_cylindrical = False,
                         z_0 = transformation_constants.Z_0, 
                         r_0 = transformation_constants.R_0,
                         debug = False):

    Z_0 = z_0
    R_0 = r_0

    assert len(df) > 0, "Error! No data found in input DataFrame!"
    assert len(df_crt) > 0, "Error! No data found in input galactocentric DataFrame!"

    if(debug):
        print("Generating covariance matrices from input data..")
        tic=timeit.default_timer()

    cov_dict = {}

    # A piece of code whose point is to prevent indexing over 
    # dataframe inside the loop. It takes too long.

    if(df_crt is not None):
        df = pd.concat([df, df_crt], axis=1)

    for row in df.itertuples():   

        if(debug):
            print("Generating covariance matrix of {0}".format(row.Index))

        # Get covariance matrix from ICRS coordinates
        C = generate_covmat(row)

        if(transform_to_galcen is True):
            C = transform_cov_matrix(C, row, "Cartesian", Z_0, R_0)
        
        # Transforms to cylindrical coordinate system. Can only be done if coordinates are in galactocentric frame.
        # Expects DF with parameters in Cartesian.

        # TODO: Implement exception handling in the future
        # EXAMPLE: If cylindrical coordinates not found give an error.
        if(transform_to_cylindrical is True):
            
            #sub_df_crt = df_crt.iloc[row.Index]
            C = transform_cov_matrix(C, row, "Cylindrical", Z_0, R_0)    

        # Append
        cov_dict[row.source_id] = C
        
    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for covariance matrix generation and transformation: {a} sec".format(a=toc-tic))

    return cov_dict

'''
A new function for transforming covariance matrices of whole data set.
The idea is that it will not create a new matrix every iteration but 
transform the initial ones generated from Gaia input data. 
'''
def transform_cov_matrices():
    pass


'''
Function that gets the covariance matrix of a specific point source (row in DataFrame).
'''
def generate_covmat(sub_df):

    # Declare empty matrix
    C = np.zeros((6, 6))
    
    # Possible parameter names
    names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    
    # For Diagonal Elements
    for i, name in enumerate(names):
            ext = names[i] + "_error"

            if not ext in sub_df._fields:
                print("{0} not in data!".format(ext))
                return

            err = getattr(sub_df, ext)       
 
            if(name == 'ra' or name == 'dec'):

                # This converts from [mas] to [deg]
                err = err/(3.6 * 10**6)

            C[i, i] = err ** 2
        
    # For Radial Velocity Element        
    C[5,5] = getattr(sub_df,'radial_velocity_error') ** 2        

    # For Rest of the Elements
    for i, name1 in enumerate(names):
                for j, name2 in enumerate(names):
                
                    if j <= i:
                        continue

                    ext = "{0}_{1}_corr".format(name1, name2)

                    if not ext in sub_df._fields:
                        print("{0} not in data!".format(ext))
                        return

                    corr = getattr(sub_df,ext)
               
                    # Sqrt because it accesses values from the main diagonal which are squared.
                    C[i, j] = corr * np.sqrt(C[i, i] * C[j, j])
                    C[j, i] = C[i, j]

    return C


def transform_cov_matrix(C, sub_df, coordinate_system, z_0 = transformation_constants.Z_0, r_0 = transformation_constants.R_0):

    # Grabs the correct Jacobian depending on the coordinate system needed
    J = transformation_constants.get_jacobian(sub_df, coordinate_system, Z_0 = transformation_constants.Z_0, R_0 = transformation_constants.R_0)

    C_transformed = J @ C @ J.T
    
    return C_transformed
