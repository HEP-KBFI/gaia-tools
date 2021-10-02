'''
File for storing functions and constants related to generating and transforming
data covariance matrices.
'''

import numpy as np
from . import transformation_constants
import pandas as pd
import timeit

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
                         is_bayes = False,
                         debug = False):

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

    # Transforms to cylindrical coordinate system. Can only be done if coordinates are in galactocentric frame.
    # Expects DF with parameters in Cartesian.
    
    # TODO: Implement exception handling in the future
    # EXAMPLE: If cylindrical coordinates not found give an error.
    # EXAMPLE: If data_array not numpy array -> Exception 
    if(transform_to_cylindrical is True):

        data_array = df_crt[["x", "y","r","phi","v_r","v_phi"]].to_numpy()

        if isinstance(data_array, np.ndarray):
            C = transform_cov_matrix(C, data_array, "Cylindrical", Z_0, R_0)
        else: 
            print("Data is not a numpy array!")
            return

    # Unpack covariance matrices to list
    # TODO: Figure out a more efficient way to do this!! 
    cov_list = list(C)
       
    d = {"source_id": df.source_id, "cov_mat": cov_list}
    cov_df = pd.DataFrame(d)

    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for covariance matrix generation and transformation: {a} sec".format(a=toc-tic))

    return cov_df

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
def generate_covmat(df):

    n = len(df)
    
    # Declare empty matrix array
    C = np.zeros((n, 6, 6))
    
    # Possible parameter names
    names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    
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


def transform_cov_matrix(C, df, coordinate_system, z_0 = transformation_constants.Z_0, r_0 = transformation_constants.R_0, is_bayes = False):

    # Grabs the correct Jacobian for every point in data set. Of shape (n, 6, 6).
    J = transformation_constants.get_jacobian(df, coordinate_system, Z_0 = transformation_constants.Z_0, R_0 = transformation_constants.R_0, is_bayes = is_bayes)
    
    J = J.T.reshape(len(df), 6, 6, order = 'A').swapaxes(1,2)

    J_trunc= J.reshape(len(df),6,6, order = 'A').swapaxes(1,2)

    C_transformed = J @ C @ J_trunc
    
    return C_transformed


'''
OLD CODE FROM UNOPTIMISED VERSION
'''

#def generate_covmatrices(df, 
#                         df_crt = None, 
#                         transform_to_galcen = False, 
#                         transform_to_cylindrical = False,
#                         z_0 = transformation_constants.Z_0, 
#                         r_0 = transformation_constants.R_0,
#                         debug = False):

#    Z_0 = z_0
#    R_0 = r_0

#    assert len(df) > 0, "Error! No data found in input DataFrame!"
#    assert len(df_crt) > 0, "Error! No data found in input galactocentric DataFrame!"

#    if(debug):
#        print("Generating covariance matrices from input data..")
#        tic=timeit.default_timer()

#    #cov_dict = {}
#    cov_list = []

#    # A piece of code whose point is to prevent indexing over 
#    # dataframe inside the loop. It takes too long.

#    if(df_crt is not None):
#        df = pd.concat([df, df_crt], axis=1)

#    for row in df.itertuples():   

#        if(debug):
#            print("Generating covariance matrix of {0}".format(row.Index))

#        # Get covariance matrix from ICRS coordinates
#        C = generate_covmat(row)

#        if(transform_to_galcen is True):
#            C = transform_cov_matrix(C, row, "Cartesian", Z_0, R_0)
        
#        # Transforms to cylindrical coordinate system. Can only be done if coordinates are in galactocentric frame.
#        # Expects DF with parameters in Cartesian.

#        # TODO: Implement exception handling in the future
#        # EXAMPLE: If cylindrical coordinates not found give an error.
#        if(transform_to_cylindrical is True):
            
#            #sub_df_crt = df_crt.iloc[row.Index]
#            C = transform_cov_matrix(C, row, "Cylindrical", Z_0, R_0)    

#        # Append
#        #cov_dict[row.source_id] = C
#        cov_list.append((row.source_id, C))
       
#    cov_df = pd.DataFrame(cov_list, columns=['source_id', 'cov_mat'])

#    if(debug):
#        toc=timeit.default_timer()
#        print("Time elapsed for covariance matrix generation and transformation: {a} sec".format(a=toc-tic))

#    return cov_df

#'''
#A new function for transforming covariance matrices of whole data set.
#The idea is that it will not create a new matrix every iteration but 
#transform the initial ones generated from Gaia input data. 
#'''
#def transform_cov_matrices():
#    pass


#'''
#Function that gets the covariance matrix of a specific point source (row in DataFrame).
#'''
#def generate_covmat(sub_df):

#    # Declare empty matrix
#    C = np.zeros((6, 6))
    
#    # Possible parameter names
#    names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    
#    # For Diagonal Elements
#    for i, name in enumerate(names):
#            ext = names[i] + "_error"

#            if not ext in sub_df._fields:
#                print("{0} not in data!".format(ext))
#                return

#            err = getattr(sub_df, ext)       
 
#            if(name == 'ra' or name == 'dec'):

#                # This converts from [mas] to [deg]
#                err = err/(3.6 * 10**6)

#            C[i, i] = err ** 2
        
#    # For Radial Velocity Element        
#    C[5,5] = getattr(sub_df,'radial_velocity_error') ** 2        

#    # For Rest of the Elements
#    for i, name1 in enumerate(names):
#                for j, name2 in enumerate(names):
                
#                    if j <= i:
#                        continue

#                    ext = "{0}_{1}_corr".format(name1, name2)

#                    if not ext in sub_df._fields:
#                        print("{0} not in data!".format(ext))
#                        return

#                    corr = getattr(sub_df,ext)
               
#                    # Sqrt because it accesses values from the main diagonal which are squared.
#                    C[i, j] = corr * np.sqrt(C[i, i] * C[j, j])
#                    C[j, i] = C[i, j]

#    return C


#def transform_cov_matrix(C, sub_df, coordinate_system, z_0 = transformation_constants.Z_0, r_0 = transformation_constants.R_0):

#    # Grabs the correct Jacobian depending on the coordinate system needed
#    J = transformation_constants.get_jacobian(sub_df, coordinate_system, Z_0 = transformation_constants.Z_0, R_0 = transformation_constants.R_0)

#    C_transformed = J @ C @ J.T
    
#    return C_transformed

