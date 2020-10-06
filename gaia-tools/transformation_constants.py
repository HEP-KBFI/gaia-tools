'''
File for storing different constants and requisite matrices for transforming coordinates to 
a different frame of reference.
'''

import numpy as np

# Distance to the galactic centre in pc
R_GALCEN = 8178

# Distance above galactic midplane in pc
Z_0 = 27

# Angle theta
THETA_SUN = np.arcsin(Z_0/R_GALCEN)

# Tangential velocity of the Sun. Currently borrowed values from Astropy!
V_SUN = np.array([[11.1], 
                  [232.24], 
                  [7.25]])


'''
A transposed matrix, which depends on the values of the ICRS coordinates of the north galactic pole and the 
galactic longitude of the first intersection of the galactic plane with the equator.
For more information see p. 165 in the Gaia DR2 documentation.
'''
A = np.array([[(-1)*0.0548755604162154, (-1)*0.8734370902348850, (-1)*0.4838350155487132],
            [0.4941094278755837, (-1)*0.4448296299600112, 0.7469822444972189],
            [(-1)*0.8676661490190047, (-1)*0.1980763734312015, 0.4559837761750669]])

# Matrix H which accounts for the height of the Sun (Z_0) aboce the Galactic plane.
H = np.array([[np.cos(THETA_SUN), 0, np.sin(THETA_SUN)],
             [0, 1, 0],
             [-np.sin(THETA_SUN), 0, np.cos(THETA_SUN)]])

# Constant used for fixing the order of magnitude of distance
k1 = 10**3

# Constant for converting 1/yr to km/s
k2 = 4.74047

def get_b_matrix(ra, dec):

    B = np.array([[np.cos(ra)*np.cos(dec), -np.sin(ra), -np.cos(ra)*np.sin(dec)],
             [np.sin(ra)*np.cos(dec), np.cos(ra), -np.sin(ra)*np.sin(dec)],
             [np.sin(dec), 0, np.cos(dec)]])
    return B

