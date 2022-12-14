import numpy as np

'''
Data object for data entries with the same bin index.

Inside this class we can define functions and variables to calculate
needed error parameters that can be accessed at any later time.

Instance variables
------------------
data : DataFrame
        A subset of data from original DataFrame belonging to particular bin.

bin_num : int
        Number of the particular bin.

N_points : int
        Number of data points in particular bin.

x_boundaries : tuple()
        A tuple of x-coordinate values defining the edges of the bins in the x-direction.

y_boundaries : tuple()
        A tuple of y-coordinate values defining the edges of the bins in the y-direction.

z_boundaries : tuple()
        A tuple of z-coordinate values defining the edges of the bins in the z-direction.

'''

class Bin:
    def __init__(self, data_subset):
        self.data = data_subset

        if("Bin_index" in data_subset.index):
            self.bin_num = data_subset.Bin_index.iloc[0]

        self.bin_num = data_subset.Bin_index.iloc[0]
        self.N_points = data_subset.shape[0]
        self.x_boundaries = []
        self.y_boundaries =[]
        self.z_boundaries = []
        self.r_boundaries = []
        self.z_boundaries= []
        self.MLE_sigma = None
        self.MLE_mu = None
        self.med_sig_vphi = None
        self.A_parameter = None

    '''
    To keep this function generalized, should pass a parameter describing
    what specfic value from covariance matrix is needed
    '''
    def get_error_data(self, parameter):

        # Error information for v_r is in [3][3]
        if(parameter == 'v_r'):
            value_list = [value[3][3] for value in self.data.cov_mat]

        elif(parameter =='v_phi'):
            value_list = [value[4][4] for value in self.data.cov_mat]

        else:
            value_list = [value[4][4] for value in self.data.cov_mat]

        # CHECK IF THIS IS CORRECT!
        return np.sqrt(value_list)

    def get_parameter_data(self, parameter):

        parameter_array = [value for value in self.data[parameter]]

        return parameter_array

    '''
    This functions computes the likelihood of the bin. It takes as arguments the MLE
    estimation of both the mean and variance of the bin.

    The MLE estimations of mu and sigma are computed from the BinCollection function 'GetMLEParameters'

    '''
    def get_bin_likelihood(self, debug=False):


        sig = self.MLE_sigma
        mu = self.MLE_mu
        n = self.N_points
        velocity_array = self.get_parameter_data('v_phi')

        try:
            error_array = self.get_error_data()
        except:
            if(debug):
                print("No error data was found inside bin!")
            return 0

        assert sig != None, "No variance found in bin, oh no!"
        assert mu != None, "No mean found in bin, oh no!"

        denom_array = (sig**2 + error_array**2)**(-1)

        add1 = (np.log(denom_array**(-1))).sum()

        add2 = (((velocity_array - mu)**2)*denom_array).sum()

        result = -0.5*(add1 + add2)


        return result


    def weighted_avg_and_std(self, values, weights):
            """
            Return the weighted average and standard deviation.

            values, weights -- Numpy ndarrays with the same shape.
            """
            average = np.average(values, weights=weights)

            # Fast and numerically precise:
            variance = np.average((values-average)**2, weights=weights)

            return (average, variance)

    def bootstrap_weighted_error(self, bin_vphi, bin_sig_vphi):

        data_length = len(self.data.v_phi)
        idx_list = np.arange(data_length)
        bootstrapped_means = np.zeros(100)

        for i in range(100):
            rnd_idx = np.random.choice(idx_list, replace=True, size=len(self.data.v_phi))
            test_sample = np.array(bin_vphi)[rnd_idx]
            weights = np.array(1/bin_sig_vphi)[rnd_idx]
            bootstrapped_means[i] = np.average(test_sample, weights=weights)

        conf_int = np.percentile(bootstrapped_means, [16, 84])
        return (conf_int[1] - conf_int [0])/2


    def get_likelihood_w_asymmetry(self, v_c, drop_approx = False, debug=False):
        """Compute likelihood of bin with asymmetry taken into account

        Args:
            v_c (float): Proposed V_c of bin in MCMC
            debug (bool, optional): Verbose flag. Defaults to False.

        Returns:
            float: Returns bin likelihood
        """

        bin_vphi = self.data.v_phi.to_numpy()
        bin_sig_vphi = self.data.sig_vphi.to_numpy()

        weights = 1/bin_sig_vphi
        
        # Weighted mean
        weighted_mean = np.average(bin_vphi, weights=weights)
        #weighted_avg, weighted_var = self.weighted_avg_and_std(self.data.v_phi, weights)

        # Weighted error
        weighted_error = self.bootstrap_weighted_error(bin_vphi, bin_sig_vphi)
        #avg_sig_vphi = (weighted_var)/len(self.data.v_phi)

        # Get A for asymmetric drift computation
        A = self.A_parameter

        add_1 = np.log(2*np.pi*weighted_error)

        if(drop_approx):
            A = 2*A
            v_phi_model = v_c - A/(v_c + weighted_mean)

        else:
            v_phi_model = (v_c**2 - A)/v_c

        add_2 = (weighted_mean - v_phi_model)**2/weighted_error

        if(debug):
            print("A -> {}".format(A))
            print("Add 1 -> {}".format(add_1))
            print("Add 2 -> {}".format(add_2))
            print("Asymmetric drift -> {}".format(A/v_c))

        return -0.5*(add_1 + add_2)

    def get_med_sig_vphi(self, debug):

        try:
            self.med_sig_vphi = np.median((self.get_error_data('v_phi'))**2)
        except:
            if(debug):
                print("No error data was found inside bin!")
            return 0

    def compute_A_parameter(self, h_r = 3000, h_sig=16400, debug=False):

        """Compute the A parameter containing the disk scale length and radial velocity dispersion scale length.

        Args:
            h_r (float): The disk scale length
            h_sig (float): The radial velocity dispersion scale length.
            debug (bool, optional): Verbose flag. Defaults to False.

        Returns:
            float: Returns the parameter value
        """

        rot_vel_var = np.var(self.data.v_phi, ddof=1)
        rad_vel_var = np.var(self.data.v_r, ddof=1)

        XX = rot_vel_var/rad_vel_var

        if(debug):
            print(XX)

        # R - bin center
        R = np.mean(self.r_boundaries)

        # Without 2vc approximation
        #A = rad_vel_var*(XX - 1 + R*(1/h_r + 2/h_sig))

        A = 0.5*rad_vel_var*(XX - 1 + R*(1/h_r + 2/h_sig))

        return A
