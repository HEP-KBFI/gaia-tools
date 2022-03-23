import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import reduce

sys.path.append("../gaia_tools/")
import data_analysis
import transformation_constants
import covariance_generation
from import_functions import import_data

# LIKELIHOOD SUM FUNCTION
'''
This function uses the Gaia data in ICRS:
1) Transforms it into a Galactocentric frame using the theta arguments given
2) Generates the covariance matrices (also transforms) and appends them to the Galactocentric data
3) Bins the data and generates a 'BinCollection' object
4) Iterates over the bins and computes a likelihood value for each
5) Sums the likelihood values over the bins
'''

def get_likelihood_sum(data_icrs,
                       r = transformation_constants.R_0,
                       z = transformation_constants.Z_0,
                       Usun = transformation_constants.V_SUN[0][0],
                       Vsun = transformation_constants.V_SUN[1][0],
                       num_r_bin = 10,
                       num_z_bin = 4):

    start = time.time()
    theta = (r, z, Usun, Vsun, transformation_constants.V_SUN[2][0])

    v_sun = np.array([[theta[2]],
                              [theta[3]],
                              [theta[4]]])

    # 1
    galcen_data = data_analysis.get_transformed_data(data_icrs,
                                       include_cylindrical = True,
                                       r_0 = theta[0],
                                       v_sun = v_sun,
                                       debug = True,
                                       is_bayes = True,
                                       is_source_included = True)
    # 2
    cov_df = covariance_generation.generate_covmatrices(df = data_icrs,
                                           df_crt = galcen_data,
                                           transform_to_galcen = True,
                                           transform_to_cylindrical = True,
                                           z_0 = theta[1],
                                           r_0 = theta[0],
                                           is_bayes = True,
                                           debug=True)

    galcen_data['cov_mat'] = cov_df['cov_mat']

    min_val = np.min(galcen_data.r)
    max_val = np.max(galcen_data.r)

    min_val_z = np.min(galcen_data.z)
    max_val_z = np.max(galcen_data.z)

    # 3
    bin_collection = data_analysis.get_collapsed_bins(data = galcen_data,
                                                                 theta = theta,
                                                                 BL_r_min = min_val - 1,
                                                                 BL_r_max = max_val + 1,
                                                                 BL_z_min = -1200,
                                                                 BL_z_max = 1200,
                                                                 N_bins = (num_r_bin, num_z_bin ),
                                                                 r_drift = False,
                                                                 debug = True)

    # Computes the MLE Mu and Sigma for each bin
    bin_collection.GetMLEParameters()

    # Setup likelihood array
    n = reduce(lambda x, y: x*y, bin_collection.N_bins)
    likelihood_array = np.zeros(n)

    star_count = len(bin_collection.data)

    # Keep track how many data points are used in likelihood computation
    point_count = []

    # 4
    for i, bin in enumerate(bin_collection.bins):

        likelihood_value = bin.get_bin_likelihood(debug=True)

        if(likelihood_value == 0):
            print(theta)
            val = 0

        else:
            #print(bin.N_points)
            point_count.append(bin.N_points)

            # get bin likelihood
            val = likelihood_value

            # convert chi-squared
            #val = val*(-2)/star_count

        likelihood_array[i] = val

    print("Number of points in analysis: {0}".format(np.sum(point_count)))
    print("Bin Collection data shape: {0}".format(bin_collection.data.shape))

    likelihood_sum = np.sum(likelihood_array)

    end = time.time()
    print("Likelihood time = %s" % (end - start))

    return likelihood_sum, bin_collection, likelihood_array

    # Function that generates the neccessary variables for
# plotting the profiles

def generate_plot_vars(bin_r, bin_z, parameter):

    # The varied range in x-axis
    if(parameter == "R_0"):
        x = np.linspace(6000, 12000, 10)
    elif(parameter == "U_odot"):
        x = np.linspace(0, 50, 10)
    else:
        x = np.linspace(150, 350, 10)

    # The likelihood values
    y = []

    for i, item in enumerate(x):
        print(i, item)

        if(parameter == "R_0"):
            val = get_likelihood_sum(data_icrs,
                                    r = item,
                                    num_r_bin = bin_r,
                                    num_z_bin = bin_z)[0]

        elif(parameter == "U_odot"):

            val = get_likelihood_sum(data_icrs,
                                    Usun = item,
                                    num_r_bin = bin_r,
                                    num_z_bin = bin_z)[0]

        elif(parameter == "V_odot"):
            val = get_likelihood_sum(data_icrs,
                                    Vsun = item,
                                    num_r_bin = bin_r,
                                    num_z_bin = bin_z)[0]

        print("Likelihood: {0}".format(val))
        y.append(val)

    return x, y, parameter

# The plotting function

def generate_likelihood_plot(x, y, bin_r, bin_z, parameter, save = False):

    fig = plt.figure(figsize = (8,8))
    plt.plot(x, y, '-', color='blue')
    plt.title("Likelihood dependence on ${0}$".format(parameter), pad = 45, fontdict={'fontsize': 20})
    plt.suptitle(r"({0}x{1} bins)".format(bin_r, bin_z), y=0.93, fontsize=15)
    plt.grid()

    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    if(parameter == "R_0"):
        unit = "pc"
    else:
        unit = "km/s"

    plt.xlabel(r'${0}$ [{1}]'.format(parameter, unit), fontdict={'fontsize': 18}, labelpad = 25)
    plt.ylabel('Log Likelihood',fontdict={'fontsize': 18}, labelpad = 25)
    plt.subplots_adjust(left=0.2)

    title_string = "../out/Likelihood_{0}_{1}x{2}".format(parameter, bin_r, bin_z)

    if(save):
        plt.savefig(title_string+'.png', dpi=300)

if __name__ == "__main__":

    parameter = sys.argv[1]

    path = "/hdfs/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv"
    data_icrs = import_data(path = path, debug = False)

    bin_settings = [(5,4), (10,4), (15,4), (10, 2), (20, 8)]

    for bin_r, bin_z in bin_settings:
        x, y, parameter = generate_plot_vars(bin_r, bin_z, parameter)
        generate_likelihood_plot(x, y, bin_r, bin_z, parameter, True)