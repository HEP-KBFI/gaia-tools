'''
A module containing various testing and benchmark functions.
'''

from .import_functions import import_data
from .data_analysis import *

# Temporary function for Issue no. 18
def Collapsed_Plot_Test():

    # LOAD DATA
    #region

    my_path = "astroquery_test.csv"
    
    df = import_data(path = my_path)

    #endregion

    galcen = get_transformed_data(df, include_cylindrical = True)
    print(galcen.iloc[0:5])

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

    from .data_plot import plot_collapsed_bins, display_bins
  
    plot_collapsed_bins(bins, 'v_r', mode='index')
    plot_collapsed_bins(bins, 'v_r', mode='mean')

    print(galcen.index)



# Checks our coordinate transformation against Astropy.
def Parameter_Test(df):

    from .data_plot import run_parameter_tests

    parameter_list = ["x", "y", "z", "v_x", "v_y", "v_z"]
    run_parameter_tests(df, parameter_list)


'''
Function for testing the time it takes for Astropy package to convert data in
to a galactocentric frame of reference.
'''
def astropy_timer_benchmark():
    
    import time, timeit
    tic=timeit.default_timer()

    galcen_astropy = transform_to_galcen(df)

    toc=timeit.default_timer()

    print("Time elapsed for data {a} sec".format(a=toc-tic))
    print("<!----------------------------------------------!>")



'''
MCMC loop test function.

Expects a BinCollection object as data!
df - DataFrame with ICRS data for mcmc looper object.
'''

def MCMCFunction_Test(df, data):

    # Check get_parameter_data function
    a = data.bins[65].get_parameter_data('v_phi')
    print(a)

    # Check get_error_data function
    b = data.bins[65].get_error_data()
    print(b)


    print("Start MLE!")
    data.GetMLEParameters()
    print(data.bins[65].MLE_mu)
    print(data.bins[65].MLE_sigma)
    print("Check!")

    print("Start Likelihood Check!")


    from functools import reduce
    n = reduce(lambda x, y: x*y, data.N_bins)
    likelihood_array = np.zeros(n)

    for i, bin in enumerate(data.bins):

        

        # Get Bin likelihood
        likelihood_value = bin.get_bin_likelihood()
        
        # Add to array
        likelihood_array[i] = likelihood_value
    
        # Square sum likelihoods over all bins
        likelihood_sum = np.sum(likelihood_array**2)


    print(likelihood_sum)

    print("Check!")

    print("Starting MCMC Loop")
    from .mcmc import MCMCLooper
    from . import transformation_constants
    
    # This section right here is an example how to use the module elsewhere.
    theta_0 = (transformation_constants.R_0, transformation_constants.Z_0, transformation_constants.V_SUN[0][0], transformation_constants.V_SUN[1][0], transformation_constants.V_SUN[2][0])

    looper = MCMCLooper(df, theta_0)
    result = looper.run_sampler()

    print("Check!")