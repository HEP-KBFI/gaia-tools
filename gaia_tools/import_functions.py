'''
A module to deal with everything data import.
'''

import pandas as pd
from .data_analysis import filter_distance


# TODO: Add proper description later.
# Expects a .csv or similar format. See Pandas.read_csv.
def import_data(path, distance = 32000, debug = False):
    
    if(debug):
        import time, timeit
        tic=timeit.default_timer()

    print("Start import...")
    df = pd.read_csv(path)
   
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))
    
    # TODO: Implement proper parallax handling later!
    print("Filtering entries that are further than 32 000 pc")
    df = filter_distance(df, distance)
    
    print("The dimensions of the data: (rows, columns) -> {}".format(df.shape))

    # TODO: Implement proper negative parallax handling later!
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