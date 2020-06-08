# This module contains methods for transforming Gaia data

import pandas as pd
import numpy as np
from scipy import stats
import astropy
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable

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

    #v_x
    H, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values = z,bins = 10, statistic='mean')

    #v_y
    H2, xedges2, yedges2, binnumber2 = stats.binned_statistic_2d(x, y, values = z2,bins = 10, statistic='mean')

    if(show_bins == True):
        from data_plot import display_bins

        display_bins(xedges, yedges, {'Data': H, 'Projection':'x', 'Unit':galcen_data.v_x.unit})
        display_bins(xedges, yedges, {'Data': H2, 'Projection':'y', 'Unit':galcen_data.v_y.unit})
    
    VEC_COORDS = generate_vector_mesh(xedges, yedges)

    return ({'Data':[H, H2], 
            'Projections':['x', 'y'],
            'Velocity Units':[galcen_data.v_x.unit, galcen_data.v_y.unit],
            'Distance Units':[galcen_data.x.unit, galcen_data.y.unit],
            'Vector Coordinates':VEC_COORDS})

'''
Function for finding center points of bins in binned data and then 
creating a meshgrid out of all the point coordinates. The center points of bins
are the origin points for the vectors.
'''
def generate_vector_mesh(xedges, yedges):
    vec_x = []
    vec_y = []
    vec_z = []

    # TODO: Automate this for n
    for i in range(10):
        vec_x.append((xedges[i+1]+xedges[i])/2)
        vec_y.append((yedges[i+1]+yedges[i])/2)

    # We create a meshgrid out of all the vector locations
    VEC_XX, VEC_YY = np.meshgrid(vec_x, vec_y)

    return VEC_XX, VEC_YY

def main():
    
    # For finding current module working directory
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)


    #from astroquery.gaia import Gaia
    #Gaia.login()

    #full_qualified_table_name = 'user_spoder.table_test_from_file'
    #query = 'select * from ' + full_qualified_table_name
    #job = Gaia.launch_job(query=query, dump_to_file = True, output_format='csv', output_file='user_uploaded_table_test.csv')

    #print(job)
    
    my_path = 'spectroscopic_test_table.csv'
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

    print("Transforming data to galactocentric frame...")
    galcen = transform_to_galcen(df)

    from data_plot import distribution_hist, point_density, display_bins, generate_velocity_map
    distribution_hist(galcen)
   
    #point_density(galcen, 2000)

    bin_dict = bin_data(galcen, show_bins = True)
    generate_velocity_map(bin_dict)

    print("Plotting done!")


if __name__ == "__main__":

    
    main()

