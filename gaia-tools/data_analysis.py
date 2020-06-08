# This module contains methods for transforming Gaia data

import pandas as pd
import numpy as np
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
    # Get first 10 non-header rows from our data file
    
    my_path = 'user_uploaded_table_test.csv'
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

    from data_plot import distribution_hist
    distribution_hist(galcen)
   
    print("Plotting done!")


if __name__ == "__main__":

    
    main()

