from Bin import Bin
import numpy as np


'''
A collection of spatially binned data ("Bin") structures.
'''
class BinCollection:
    def __init__(self, data):
        self.data = data
        self.bins = []
        self.N_bins = []
        self.x_boundaries = []
        self.y_boundaries =[]
        self.bin_num_set = set(data.Bin_index)
    '''
    Collect all bins into a list. Numbered from bottom to top.

    '''
    def GenerateBins(self):
        bin_index = self.N_bins + 3
        N_bins = self.N_bins
        print("------")
        print(N_bins)
        for bin_index in range(((N_bins+2)*N_bins)-2):
            data_subset = self.data[self.data.Bin_index == bin_index]
            self.bins.append(Bin(data_subset))


