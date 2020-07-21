class Bin:
    def __init__(self, data_subset):
        self.data = data_subset
        self.bin_num = data_subset.Bin_index.iloc[1]
        self.N_points = data_subset.shape[0]
        self.x_boundaries = []
        self.y_boundaries =[]


