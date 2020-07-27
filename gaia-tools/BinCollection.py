from Bin import Bin
import numpy as np
import pandas as pd

'''
A collection of spatially binned data ("Bin") structures.
'''
class BinCollection:
    def __init__(self, data, N_bins, XX, YY, debug = False):
        
        self.data = data
        self.bins = []
        self.N_bins = N_bins
        self.bin_boundaries = XX, YY
        self.bin_num_set = set(data.Bin_index)
        self.debug = debug
        
    '''
    Collect all bins provided parameters from the 'binned_statistic_2d' function

    '''
    def GenerateBins(self):
        N_bins = self.N_bins
        max_bin_index = ((N_bins+3) * N_bins)     
        bin_index = N_bins + 3
        row_count = 0
        
        while bin_index < max_bin_index + 1:
            
            # If reaches end of column will skip 2 next index numbers 
            if(row_count == N_bins):
                
                if(self.debug):
                    print("Skipping bin: {} and {}!".format(bin_index, bin_index+1))
               
                row_count = 0
                bin_index = bin_index + 2		
                continue

            # If no data is inside bin, will return Bin with empty DataFrame
            if(bin_index not in self.bin_num_set):
                
                if(self.debug):
                    print("Empty bin: {}!".format(bin_index))
                    
                columns= self.data.columns
                data_subset=pd.DataFrame(columns=columns)
                
                mock_data = []
                for col in columns:
                    mock_data.append(np.nan)
                
                data_subset.loc[0] = mock_data
                data_subset.iloc[0].Bin_index = bin_index
                
                self.bins.append(Bin(data_subset))
                
                row_count = row_count + 1
                bin_index += 1
                continue
            
            # Adds Bin object to list of bins with correct data
            data_subset = self.data[self.data.Bin_index == bin_index]
            self.bins.append(Bin(data_subset))
            row_count += 1
            bin_index += 1
            
        '''
        Assign boundaries to pre-loaded bins from bottom -> top column wise

        XX - 2D array mapping out x-coordinate values for each step of y-coordinates.
        YY - 2D array mapping out y-coordinate values for each step of x-coordinates.

        XX and YY both have shape (nrows, ncols)
        '''
        XX = self.bin_boundaries[0]
        YY = self.bin_boundaries[1]
        
        count = 0

        # For j in range of columns: -1 because we look at binwise not edgewise
        for j in range(XX.shape[1]-1):

                # For i in range of rows: again binwise
                for i in range(YY.shape[0]-1):

                    # Grabs adjacent bin edges in the x-direction
                    temp_x = (XX[0][j], XX[0][j+1])

                    # Grabs adjacent bin edges in the y-direction
                    temp_y = (YY.T[0][i], YY.T[0][i+1])

                    # Assign boundaries to bin objects
                    self.bins[count].x_boundaries = temp_x
                    self.bins[count].y_boundaries = temp_y

                    count = count + 1  
               
    '''
    Calculates a value inside each bin and returns a numpy array which can be plotted using 
    'pcolormesh' function.
    
    TODO: The idea here would be to generalise this function so you could use any arbitrary calculation 
    function inside each bin.

    parameter - DataFrame column name in the form of a string. For example: 'v_x', v_y', 'v_z'
    '''        
    def CalculateValues(self, parameter):
        
        # 2D list of values
        values = []
        
        # Bin boundaries in x- and y-directions
        XX = self.bin_boundaries[0]
        YY = self.bin_boundaries[1]
        
        count = 0

        # For j in range of columns: -1 because we look at binwise not edgewise
        for j in range(XX.shape[1]-1):
               
                # Empty list for a particular column
                rows = []
               
                # For i in range of rows: again binwise
                for i in range(YY.shape[0]-1):

                    # Value to be calculated
                    if(parameter == 'v_x'):
                        temp_val = -np.mean(self.bins[count].data[parameter])
                    
                    temp_val = np.mean(self.bins[count].data[parameter])
                    rows.append(temp_val)
                    count = count + 1  
                    
                values.append(rows)
                
        # Convert to a numpy array and return        
        return np.asarray(values)