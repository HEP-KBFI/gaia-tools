# This module contains methods for transforming Gaia data

import pandas as pd
import numpy as np
from scipy import stats
import astropy
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable
from BinCollection import BinCollection
import transformation_constants

'''
Function for filtering out entries that are further out than some specified distance in pc
'''
def filter_distance(df, dist, *args, **kwargs):
    
    #TODO: Assert parallax_lower type and value
    df['distance'] = 1/df.parallax

    # Distance given in pc
    df = df[df.distance <= dist]
    df.reset_index(inplace=True, drop=True)

    return df


'''
Generalised filtering function
'''
# TODO: Function with a passible dictionary of parameter restrictions
def filter_value(df, *args, **kwargs):

    #TODO: Use inplace functions from pandas
    pass


'''
Helper function for 'get_SkyCoord_object' function. Returns QTable containing parallax values.
'''
def get_parallax_table(parallax_series):

    df_parallax = pd.DataFrame(parallax_series, columns="index parallax".split())
    df_parallax.drop('index',axis=1, inplace=True)
    t = QTable.from_pandas(df_parallax)

    return t

'''
Function for instantiating a SkyCoord object with given data.
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
Function for transforming data into a galactocentric reference frame using SkyCoord objects and 
the 'transform_to' function.

Returns a SkyCoord object
'''
def transform_to_galcen(df, z_sun=17*u.pc, galcen_distance=8.178*u.kpc):
    
    c = get_SkyCoord_object(df)
    c = c.transform_to(coord.Galactocentric(z_sun=z_sun, galcen_distance=galcen_distance))

    return c

# TODO: Implement binning in all projections -> x, y, z. (Currently only on x-y plane)
'''
Function for binning data in 2 dimensions. Used to plot vector maps of velocities.
Input parameters:
    BL - Bin Limit (The edges of the xyz boundary)
'''
def bin_data(galcen_data, show_bins = False, BL = 20000):
   
    
    # DEPRECATED
    # Map values to temporary data frame.
    #plottable_df = pd.DataFrame({'x': galcen_data.x.value,
    #                        'y':galcen_data.y.value,
    #                        'z':galcen_data.z.value,
    #                        'v_x':galcen_data.v_x.value,
    #                        'v_y':galcen_data.v_y.value,
    #                        'v_z':galcen_data.v_z.value})

    # Fix for newly developed method
    plottable_df = galcen_data

    # Define spatial limits.
    plottable_df = plottable_df[(plottable_df.x >= -BL) & (plottable_df.x <= BL)]
    plottable_df = plottable_df[(plottable_df.y >= -BL) & (plottable_df.y <= BL)]
    plottable_df = plottable_df[(plottable_df.z >= -BL) & (plottable_df.z <= BL)]

    # x, y coordinates of the points
    x = -plottable_df.x
    y = plottable_df.y

    # Velocity projections of points
    z = -plottable_df.v_x
    z2 = plottable_df.v_y

    # Number of bins along main axis
    bins = (10, 10)

    # Calling the actual binning function
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, bins = bins, statistic='mean')

    # Create a meshgrid from the vertices   
    XX, YY = np.meshgrid(xedges, yedges)

    ZZ = (np.min(plottable_df.z), np.max(plottable_df.z))

    # Assign a binnumber for each data entry
    plottable_df['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(plottable_df, bins, XX, YY, ZZ)

    # Generate the bins with respective x-y boundaries
    bin_collection.GenerateBins()

    # TODO: Generalise this!
    if(show_bins == True):
        from data_plot import display_bins

        display_bins(bin_collection, 'v_x')
        display_bins(bin_collection, 'v_y')
        
    return bin_collection

'''
Returns bins in terms of R - Z. (Cylindrical)

data - dataframe with data
N_bins - number of bins in R direction
XX, YY, ZZ - spatial boundaries in the form: [-x ; +x], [-y ; +y], [-z ; +z],
'''
def get_collapsed_bins(data, BL_r, BL_z, N_bins = (10, 10)):
    
    # This assertion doesnt make sense, fix it later 
    assert len(data.shape) > 0, "No data!"

    if not 'r' or 'phi' in data.index:
        print("No cylindrical coordinates found!")
        return

    # Fix for newly developed method
    plottable_df = data

    # Define spatial limits.
    plottable_df = plottable_df[(plottable_df.r >= -BL_r) & (plottable_df.r <= BL_r)]
    plottable_df = plottable_df[(plottable_df.z >= -BL_z) & (plottable_df.z <= BL_z)]

    # r and z parameters of points loaded into Series
    r = plottable_df.r
    z = plottable_df.z

    # Velocity projections of points: NOT NEEDED
    c = plottable_df.v_phi

    # Calling the actual binning function
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(r, z, values = c, range = [[0, BL_r], [-BL_z, BL_z]], bins=N_bins, statistic='mean')

    # Create a meshgrid from the vertices: X, Y -> R, Z
    XX, YY = np.meshgrid(xedges, yedges)
    
    # Assign a binnumber for each data entry
    plottable_df['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(plottable_df, N_bins, XX, YY, YY, mode='r-z', debug=True)
    
    # Generate the bins with respective r-z boundaries
    bin_collection.GenerateBins()
    
    return bin_collection

    


'''
Function for finding center points of bins in binned data and then 
creating a meshgrid out of all the point coordinates. The center points of bins
are the origin points for the vectors.
'''
def generate_vector_mesh(XX, YY):
    vec_x = []
    vec_y = []
    vec_z = []

    for i in range(XX.shape[0]-1):
        vec_x.append((XX[0][i+1]+XX[0][i])/2)
        vec_y.append((YY.T[0][i+1]+YY.T[0][i])/2)

    # We create a meshgrid out of all the vector locations
    VEC_XX, VEC_YY = np.meshgrid(vec_x, vec_y)
    
    return VEC_XX, VEC_YY



#region Manual Coordinate Transformation
'''
z_0 - the distance above galactic midplane in pc
r_0 - the distance to the galctic centre in pc
v_sun - velocity vector of the Sun. Of type 3x1 np.array

z_0, r_0, v_sun default to Astropy values if not given to function explicitly

'''
def get_transformed_data(df, 
                         include_cylindrical = False, 
                         z_0 = transformation_constants.Z_0, 
                         r_0 = transformation_constants.R_0,
                         v_sun = transformation_constants.V_SUN):

    if(include_cylindrical):
         galcen_df = pd.DataFrame(columns="x y z v_x v_y v_z r phi v_r v_phi".split())
    else:
        galcen_df = pd.DataFrame(columns="x y z v_x v_y v_z".split())

    #region Loop over all data points
    for i in range(df.shape[0]):

        # Coordinate vector in galactocentric frame in xyz
        coords = transform_coordinates_galactocentric(df.ra.iloc[i], 
                                                      df.dec.iloc[i], 
                                                      df.parallax.iloc[i], 
                                                      z_0, 
                                                      r_0)

        # Velocity vector in galactocentric frame in xyz
        velocities = transform_velocities_galactocentric(df.ra.iloc[i], 
                                                         df.dec.iloc[i], 
                                                         df.parallax.iloc[i], 
                                                         df.pmra.iloc[i], 
                                                         df.pmdec.iloc[i], 
                                                         df.radial_velocity.iloc[i], 
                                                         z_0, 
                                                         r_0, 
                                                         v_sun)


        galcen_df = galcen_df.append({'x' : coords[0][0], 
                                      'y' : coords[1][0], 
                                      'z' : coords[2][0],
                                      'v_x' : velocities[0][0], 
                                      'v_y' : velocities[1][0], 
                                      'v_z' : velocities[2][0]},  
                                      ignore_index = True)

        if(include_cylindrical):

            phi = galcen_df.y[i]/galcen_df.x[i]
            vel_cyl = transform_velocities_cylindrical(velocities, phi)


            galcen_df['r'].loc[i] = np.sqrt(galcen_df.x[i]**2 + galcen_df.y[i]**2)
            galcen_df['phi'].loc[i] = np.arctan(phi)
            galcen_df['v_r'].loc[i] = vel_cyl[0][0]
            galcen_df['v_phi'].loc[i] = vel_cyl[1][0]
    #endregion

    # Returns transformed data as Pandas DataFrame   
    return galcen_df

def transform_coordinates_galactocentric(ra, dec, w, z_0, r_0):

    #TODO: Add ASSERT checks on function input parameters.
    # ra dec can only be in a specific range

    # Going from DEG -> RAD
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # from kpc -> pc
    k1 = transformation_constants.k1

    # Initial cartesian coordinate vector in ICRS
    coordxyz_ICRS = np.array([[(k1/w)*np.cos(ra)*np.cos(dec)],
                      [(k1/w)*np.sin(ra)*np.cos(dec)],
                       [(k1/w)*np.sin(dec)]])

    # Using M1, M2, M3 for transparency in case of bugs
    M1 = transformation_constants.A @ coordxyz_ICRS
    M2 = M1 - np.array([[r_0], 
                        [0], 
                        [0]])
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2

    return M3

def transform_velocities_galactocentric(ra, dec, w, mu_ra, mu_dec, v_r, z_0, r_0, v_sun):
    
    # Going from DEG -> RAD
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # from 1/yr -> km/s
    k2 = transformation_constants.k2

    # Initial velocity vector in ICRS in units km/s
    v_ICRS = np.array([[v_r],
                      [(k2/w)*mu_ra],
                      [(k2/w)*mu_dec]])

    B = transformation_constants.get_b_matrix(ra, dec)

    # Using M1, M2, M3, .. for transparency in case of bugs
    M1 = B @ v_ICRS
    M2 = transformation_constants.A @ M1
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2
    M4 = M3 + v_sun
    return M4

def transform_velocities_cylindrical(velocities, phi):

    v_cylindrical = transformation_constants.get_cylindrical_velocity_matrix(phi) @ velocities

    return v_cylindrical
#endregion





def main():
    
    # For finding current module working directory
    #import os 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(dir_path)

    #region Import Section

    # YOUR DATA FILE
    my_path = "astroquery_test.csv"
    
    print("Start import...")
    df = pd.read_csv(my_path)
   
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))
    
    print("Filtering entries that are further than 32 000 pc")
    df = filter_distance(df, 32000)
    
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))

    print("Removing negative parallaxes...")
    df=df[df.parallax > 0]

    df.reset_index(inplace=True, drop=True)
    print("Checking indexing...")
    print(df.head)

    #endregion File Import Section


    


    print("Transforming data to galactocentric frame...")
    
    # Our Method
    galcen2 = get_transformed_data(df, include_cylindrical = True)

    from data_plot import distribution_hist, point_density_histogram, display_bins, generate_velocity_map, run_parameter_tests
 
    import covariance_generation as cov
    import time, timeit

    #tic=timeit.default_timer()

    #cov_dict = cov.generate_covmatrices(df, df_crt = galcen2, transform_to_galcen = True, transform_to_cylindrical = True)
    
    #toc=timeit.default_timer()
    #print("Time elapsed {a} sec".format(a=toc-tic))
    #print("Covariance matrices...")

    #print(cov_dict)

    print(galcen2)
    bins = bin_data(galcen2,  show_bins = True)

    display_bins(bins, projection_parameter = 'v_x', mode='index')
    
    #generate_velocity_map(bins)

    print("The data is from a galactic slice of height: {0}".format(bins.bins[0].z_boundaries))
     
    
    print("Plotting done!")

# Temporary function for Issue no. 18
def Collapsed_Plot_Test():

    # LOAD DATA
    #region

    my_path = "astroquery_test.csv"
    
    print("Start import...")
    df = pd.read_csv(my_path)
   
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))
    
    print("Filtering entries that are further than 32 000 pc")
    df = filter_distance(df, 32000)
    
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))

    print("Removing negative parallaxes...")
    df=df[df.parallax > 0]

    df.reset_index(inplace=True, drop=True)
    print("Checking indexing...")
    print(df.head)

    #endregion

    Parameter_Test(df)

    galcen = get_transformed_data(df, include_cylindrical = True)
    print(galcen.iloc[0:5])

    print("Data Loaded Successfully.")

    bins = get_collapsed_bins(galcen, 100000, 5000, N_bins = (5, 10))
     
    #Testing bin method manually
    temp = []
    for index, row in galcen.iterrows():
        
        if(row.r >= 0 and row.r < 10000 and row.z >= 0 and row.z < 1000):
            temp.append(row.v_phi)

    mean = np.mean(temp)
    print(mean)

    print(bins.bins)
    print(bins.bins[17].data)

    from data_plot import plot_collapsed_bins, display_bins

    
    plot_collapsed_bins(bins, 'v_r', mode='mean')

    print(galcen.index)


# Move this to separate test module later
def Parameter_Test(df):

    from data_plot import run_parameter_tests

    parameter_list = ["x", "y", "z", "v_x", "v_y", "v_z"]
    run_parameter_tests(df, parameter_list)


if __name__ == "__main__":

    #main()
    Collapsed_Plot_Test()

