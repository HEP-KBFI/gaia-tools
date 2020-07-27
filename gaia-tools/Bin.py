'''
Data object for data entries with the same bin index.

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
'''

class Bin:
    def __init__(self, data_subset):
        self.data = data_subset
        self.bin_num = data_subset.Bin_index.iloc[0]
        self.N_points = data_subset.shape[0]
        self.x_boundaries = []
        self.y_boundaries =[]


