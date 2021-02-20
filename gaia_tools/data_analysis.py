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

NOTE! Distance from Solar System!
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

# TODO: Implement binning in all projections -> x, y, z. (Currently only on x-y plane)
'''
Function for binning data in 2 dimensions. Used to plot vector maps of velocities.
Input parameters:
    BL - Bin Limit (The edges of the xyz boundary)
'''
def bin_data(galcen_data, show_bins = False, BL = 20000, N_bins = (10, 10)):
   
    
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

    # Calling the actual binning function
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z, bins = N_bins, statistic='mean')

    # Create a meshgrid from the vertices   
    XX, YY = np.meshgrid(xedges, yedges)

    ZZ = (np.min(plottable_df.z), np.max(plottable_df.z))

    # Assign a binnumber for each data entry
    plottable_df['Bin_index'] = binnumber

    # Instantiate a BinCollection object
    bin_collection = BinCollection(plottable_df, N_bins, XX, YY, ZZ)

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

    for i in range(XX.shape[1]-1):
        vec_x.append((XX[0][i+1]+XX[0][i])/2)

    for j in range(YY.shape[0]-1):
        vec_y.append((YY.T[0][j+1]+YY.T[0][j])/2)

    # We create a meshgrid out of all the vector locations
    VEC_XX, VEC_YY = np.meshgrid(vec_x, vec_y)
    
    return VEC_XX, VEC_YY



#region Manual Coordinate Transformation
'''
Main function for transforming coordinates to galactocentric frame of reference.

z_0 - the distance above galactic midplane in pc
r_0 - the distance to the galctic centre in pc
v_sun - velocity vector of the Sun. Of type 3x1 np.array

z_0, r_0, v_sun default to Astropy values if not given to function explicitly

'''
def get_transformed_data(df, 
                         include_cylindrical = False, 
                         z_0 = transformation_constants.Z_0, 
                         r_0 = transformation_constants.R_0,
                         v_sun = transformation_constants.V_SUN,
                         debug = False,
                         is_source_included = False):

    if(debug):
        import timeit, time
        tic=timeit.default_timer()
    
        print("Starting galactocentric transformation loop over all data points.. ")

    #region Loop over all data points

    coords_list = []
    velocities_list = []
    coords_cyl_list = []
    velocities_cyl_list = []

    for row in df.itertuples():

        
        if(debug):
            print("Finding coordinates of {0}".format(row.Index))

        # Coordinate vector in galactocentric frame in xyz
        coords = transform_coordinates_galactocentric(row.ra, 
                                                      row.dec, 
                                                      row.parallax, 
                                                      z_0, 
                                                      r_0)

        coords_list.append(coords)

        if(debug):
            print("Finding velocity of {0}".format(row.Index))

        # Velocity vector in galactocentric frame in xyz
        velocities = transform_velocities_galactocentric(row.ra, 
                                                         row.dec, 
                                                         row.parallax, 
                                                         row.pmra, 
                                                         row.pmdec, 
                                                         row.radial_velocity, 
                                                         z_0, 
                                                         r_0, 
                                                         v_sun)
        velocities_list.append(velocities)
        
        
        if(include_cylindrical):

            phi = coords_list[row.Index][1]/coords_list[row.Index][0]
            vel_cyl = transform_velocities_cylindrical(velocities, phi)

            coords_cyl_list.append( (np.sqrt(coords_list[row.Index][0]**2 + coords_list[row.Index][1]**2), np.arctan(phi)))

            velocities_cyl_list.append( (vel_cyl[0], vel_cyl[1]))
    
    #endregion

        
    
    coords_df = pd.DataFrame(coords_list, columns="x y z".split())
    velocities_df = pd.DataFrame(velocities_list, columns="v_x v_y v_z".split())

    galcen_df = pd.concat([coords_df, velocities_df], axis=1)

    if(include_cylindrical):
        coords_df = pd.DataFrame(coords_cyl_list, columns="r phi".split())
        velocities_df = pd.DataFrame(velocities_cyl_list, columns="v_r v_phi".split())
        df_1 = pd.concat([coords_df, velocities_df], axis=1)
        galcen_df = pd.concat([galcen_df, df_1], axis=1)
     
    if(is_source_included):

        if not 'source_id' in df.columns:
            print("Error! Source ID column not found in input DataFrame!")
        
        galcen_df['source_id'] = df.source_id

       
    if(debug):
        toc=timeit.default_timer()
        print("Time elapsed for data coordinate transformation: {a} sec".format(a=toc-tic))
    
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

    # Declaring constants to reduce process time
    c1 = k1/w
    cosdec = np.cos(dec)

    # Initial cartesian coordinate vector in ICRS
    coordxyz_ICRS = np.array([[(c1)*np.cos(ra)*cosdec],
                      [(c1)*np.sin(ra)*cosdec],
                       [(c1)*np.sin(dec)]])

    # Using M1, M2, M3 for transparency in case of bugs
    M1 = transformation_constants.A @ coordxyz_ICRS
    M2 = M1 - np.array([[r_0], 
                        [0], 
                        [0]])
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2

    result = (M3[0][0], M3[1][0], M3[2][0])

    return result

def transform_velocities_galactocentric(ra, dec, w, mu_ra, mu_dec, v_r, z_0, r_0, v_sun):
    
    # Going from DEG -> RAD
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    # from 1/yr -> km/s
    k2 = transformation_constants.k2

    # Declaring constants to reduce process time
    c2 = k2/w

    # Initial velocity vector in ICRS in units km/s
    v_ICRS = np.array([[v_r],
                      [(c2)*mu_ra],
                      [(c2)*mu_dec]])

    B = transformation_constants.get_b_matrix(ra, dec)

    # Using M1, M2, M3, .. for transparency in case of bugs
    M1 = B @ v_ICRS
    M2 = transformation_constants.A @ M1
    M3 = transformation_constants.get_H_matrix(z_0, r_0) @ M2
    M4 = M3 + v_sun

    result = (M4[0][0], M4[1][0], M4[2][0])

    return result

def transform_velocities_cylindrical(velocities, phi):

    v_cylindrical = transformation_constants.get_cylindrical_velocity_matrix(phi) @ velocities

    return v_cylindrical
#endregion

def main():
    from data_plot import distribution_hist, point_density_histogram, display_bins, generate_velocity_map, run_parameter_tests
    import covariance_generation as cov
    import time, timeit
    
    # For finding current module working directory
    #import os 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(dir_path)

    # YOUR DATA FILE
    my_path = "astroquery_test.csv"
    full_path = r"C:\Users\SvenP\Desktop\Gaia Tools Project\Notebooks\Spectroscopic_Data_With_Correlations.csv"


    df = import_data(path = my_path, debug = True)

    galcen2 = get_transformed_data(df, include_cylindrical = True, debug = True, is_source_included = True)
    
    print("\n",galcen2)

    # Astropy Time Benchmark
    #tic=timeit.default_timer()
    #galcen_astropy = transform_to_galcen(df)
    #toc=timeit.default_timer()
    #print("Time elapsed for data {a} sec".format(a=toc-tic))
    #print("*******")
    
    cov_dict = cov.generate_covmatrices(df, df_crt = galcen2, transform_to_galcen = False, transform_to_cylindrical = True, debug=True)

    return;
    

    print("START PRINT")
    
    print(galcen2)
    bins = bin_data(galcen2, show_bins = True, N_bins = (10, 10))

    display_bins(bins, projection_parameter = 'v_x', mode='index')
    
    generate_velocity_map(bins)

    #print("The data is from a galactic slice of height: {0}".format(bins.bins[0].z_boundaries))
    print("END OF MAIN")

# Temporary function for Issue no. 18
def Collapsed_Plot_Test():

    # LOAD DATA
    #region

    my_path = "astroquery_test.csv"
    
    df = import_data(path = my_path)

    #endregion

    galcen = get_transformed_data(df, include_cylindrical = True)
    #print(galcen.iloc[0:5])

    print("Data Loaded Successfully.")

    bins = get_collapsed_bins(galcen, 100000, 5000, N_bins = (10, 10))
     
    #Testing bin method manually
    #temp = []
    #for index, row in galcen.iterrows():
        
    #    if(row.r >= 0 and row.r < 10000 and row.z >= 0 and row.z < 1000):
    #        temp.append(row.v_phi)

    #mean = np.mean(temp)
    #print(mean)

    #print(bins.bins)
    #print(bins.bins[17].data)

    from data_plot import plot_collapsed_bins, display_bins

    
    plot_collapsed_bins(bins, 'v_r', mode='mean')

    print(galcen.index)


# Move this to separate test module later
def Parameter_Test(df):

    from data_plot import run_parameter_tests

    parameter_list = ["x", "y", "z", "v_x", "v_y", "v_z"]
    run_parameter_tests(df, parameter_list)

# Move this to separate import module later
def import_data(path, distance = 32000, debug = False):
    
    if(debug):
        import time, timeit
        tic=timeit.default_timer()

    print("Start import...")
    df = pd.read_csv(path, nrows=100)
   
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))
    
    print("Filtering entries that are further than 32 000 pc")
    df = filter_distance(df, distance)
    
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))

    print("Removing negative parallaxes...")
    df=df[df.parallax > 0]

    df.reset_index(inplace=True, drop=True)
    print("Checking indexing... \n")

    if(debug):
        print(df.head, '\n')
        toc=timeit.default_timer()
        print("Time elapsed for data import: {a} sec".format(a=toc-tic))
        print("<!--------------------------------------------------!> \n")

    return df


if __name__ == "__main__":

    main()
    #Collapsed_Plot_Test()

