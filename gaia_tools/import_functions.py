'''
A module to deal with everything data import.
'''

import pandas as pd
from .data_analysis import filter_distance


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

    if(!is_bayes):
       
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