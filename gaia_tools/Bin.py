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
        self.MLE_sigma = null
        self.MLE_mu = null

    '''
    To keep this function generalized, should pass a parameter describing
    what specfic value from covariance matrix is needed
    '''
    def get_error_data(parameter = null):

        # Something like this
        #value_array = [value for valie in self.data.covmat[parameter]

        return []

    def get_parameter_data(parameter = null):

        # Currently specified to v_phi
        return self.data.v_phi

    '''
    This functions computes the likelihood of the bin. It takes as arguments the MLE
    estimation of both the mean and variance of the bin.

    The MLE estimations of mu and sigma are computed from the BinCollection function 'GetMLEParameters'

    '''
    def get_bin_likelihood():
    
        
        sig = self.MLE_sigma
        mu = self.MLE_mu
        n = self.N_points
        velocity_array = self.data.v_phi
        error_array = get_error_data()

        assert sig != null, "No variance found in bin, oh no!"
        assert mu != null, "No mean found in bin, oh no!"

        def sum_func(i):
    
            sum_result = np.log(sig**2 + error_array[i]) - ((velocity_array[i] - mu)**2)/((sig**2 + error_array[i]))
            
            return sum_result
    
        result = -0.5 * np.array([sum_func(i) for i in range(0, n)]).sum()
    
        return result
        
        


