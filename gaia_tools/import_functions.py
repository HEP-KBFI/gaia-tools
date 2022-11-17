'''
A module to deal with everything data import.
'''

import pandas as pd
from data_analysis import filter_distance
import sys
sys.path.append("../gaia_tools/")
sys.path.append("../scripts/")
import data_analysis
import covariance_generation as cov
import transformation_constants
import photometric_cut
import numpy as np


'''
The main import function which expects a .csv file with Gaia data.

path - data file path
distance - distance in pc to filter out stars at some specific distance
is_bayes - True if data contains Bayes inferenced distances
filter_distance - enable/disable Filtering
debug - print import duration and other info

'''
# Expects a .csv or similar format. See Pandas.read_csv.
def import_data(path, distance = 32000, is_bayes = True, filter_distance = False, test_run = False, debug = False):

    if(debug):
        import time, timeit
        tic=timeit.default_timer()


    if(test_run):
        print("Start import...")
        df = pd.read_csv(path, nrows = 100)

    else:
        print("Start import...")
        df = pd.read_csv(path)

    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))


    if(filter_distance):

        print("Filtering entries that are further than 32 000 pc")
        df = filter_distance(df, distance)

        print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))

    if(is_bayes == False):

        print("Removing negative parallaxes...")
        df=df[df.parallax > 0]

    # Reset index to start from 0.
    df.reset_index(inplace=True, drop=True)
    print("Checking indexing... \n")

    if(debug):
        print(df.head, '\n')
        toc=timeit.default_timer()
        print("Time elapsed for data import: {a} sec".format(a=toc-tic))
        print("<!--------------------------------------------------!> \n")

    return df



def import_baseline_sample():
    
    # Create outpath for current run
    run_out_path = "/home/svenpoder/repos/gaia-tools/jupyter-notebook"

    print("Photometric cut..")
    sample_IDs = photometric_cut.get_sample_IDs(run_out_path, 0.3, False)

    # The path containing the initial ICRS data with Bayesian distance estimates.
    my_path = "/home/svenpoder/DATA/Gaia_2MASS Data_DR2/gaia_rv_data_bayes.csv"

    # Import ICRS data
    icrs_data = import_data(path = my_path, is_bayes = True, debug = True)
    icrs_data = icrs_data.merge(sample_IDs, on='source_id', suffixes=("", "_y"))
    icrs_data.reset_index(inplace=True, drop=True)

    print("Size of sample after diagonal cut in ROI {}".format(icrs_data.shape))

    ## TRANSFORMATION CONSTANTS
    v_sun = transformation_constants.V_SUN
    z_0 = transformation_constants.Z_0
    r_0 = transformation_constants.R_0

    galcen_data = data_analysis.get_transformed_data(icrs_data,
                                        include_cylindrical = True,
                                        z_0 = z_0,
                                        r_0 = r_0,
                                        v_sun = v_sun,
                                        debug = True,
                                        is_bayes = True,
                                        is_source_included = True)

    galactocentric_cov = cov.generate_galactocentric_covmat(icrs_data, True)
    cyl_cov = cov.transform_cov_cylindirical(galcen_data, galactocentric_cov)
    galcen_data = galcen_data.merge(cyl_cov, on='source_id')


    # Final data selection
    galcen_data = galcen_data[(galcen_data.r < 12000) & (galcen_data.r > 5000)]
    galcen_data = galcen_data[(galcen_data.z < 200) & (galcen_data.z > -200)]
    galcen_data.reset_index(inplace=True, drop=True)
    print("Final size of sample {}".format(galcen_data.shape))

    icrs_data = icrs_data.merge(galcen_data, on='source_id')[icrs_data.columns]

    min_r = np.min(galcen_data.r)
    max_r = np.max(galcen_data.r)

    return galcen_data, min_r, max_r

