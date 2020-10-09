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
   
    

    # Map values to temporary data frame.
    plottable_df = pd.DataFrame({'x': galcen_data.x.value,
                            'y':galcen_data.y.value,
                            'z':galcen_data.z.value,
                            'v_x':galcen_data.v_x.value,
                            'v_y':galcen_data.v_y.value,
                            'v_z':galcen_data.v_z.value})

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
    bins = 10

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
        from data_plot import display_mean_velocity

        display_mean_velocity(bin_collection, 'v_x')
        display_mean_velocity(bin_collection, 'v_y')
        
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

def get_transformed_data(df):

    galcen_df = pd.DataFrame(columns="x y z v_x v_y v_z".split())

    for i in range(df.shape[0]):

        # Coordinate vector in galactocentric frame in xyz
        coords = transform_coordinates_galactocentric(df.ra.iloc[i], df.dec.iloc[i], df.parallax.iloc[i])

        # Velocity vector in galactocentric frame in xyz
        velocities = transform_velocities_galactocentric(df.ra.iloc[i], df.dec.iloc[i], df.parallax.iloc[i], df.pmra.iloc[i], df.pmdec.iloc[i], df.radial_velocity.iloc[i])


        galcen_df = galcen_df.append({'x' : coords[0][0], 
                              'y' : coords[1][0], 
                              'z' : coords[2][0],
                              'v_x' : velocities[0][0], 
                              'v_y' : velocities[1][0], 
                              'v_z' : velocities[2][0]},  
                ignore_index = True)

    return galcen_df

def transform_coordinates_galactocentric(ra, dec, w):

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
    M2 = M1 - np.array([[transformation_constants.R_GALCEN], 
                        [0], 
                        [0]])
    M3 = transformation_constants.H @ M2

    return M3

def transform_velocities_galactocentric(ra, dec, w, mu_ra, mu_dec, v_r):
    
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
    M3 = transformation_constants.H @ M2
    M4 = M3 + transformation_constants.V_SUN
    return M4

#endregion





def main():
    
    # For finding current module working directory
    #import os 
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #print(dir_path)

    # START File Import Section

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

    # END File Import Section

    # Testing Our Results to Astropy Functions
    print("Transforming data to galactocentric frame...")
    
    # Old Method
    galcen = transform_to_galcen(df)
    print(galcen[0:5])

    # Our Method
    galcen2 = get_transformed_data(df)
    print(galcen2.iloc[0:5])

    from data_plot import distribution_hist, point_density_histogram, display_mean_velocity, generate_velocity_map
    #distribution_hist(galcen)
   
    point_density_histogram(galcen, 50)
    point_density_histogram(galcen2, 50)

    import covariance_generation as cov

    cov_dict = cov.generate_covmatrices(df)

    print(cov_dict)


    #bins = bin_data(galcen, show_bins = True)
    #generate_velocity_map(bins)

    #print("The data is from a galactic slice of height: {0}".format(bins.bins[0].z_boundaries))
    

    #print("Plotting done!")


if __name__ == "__main__":

    main()

