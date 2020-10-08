'''
File for storing different constants and requisite matrices for transforming coordinates to 
a different frame of reference.
'''

import numpy as np

#region CONSTANTS

# Distance to the galactic centre in pc
R_GALCEN = 8178

# Distance above galactic midplane in pc
Z_0 = 27

# Angle theta (in rad)
THETA_SUN = np.arcsin(Z_0/R_GALCEN)

# Velocity vector of the Sun. Currently borrowed values from Astropy!
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

#endregion

def get_b_matrix(ra, dec):

    B = np.array([[np.cos(ra)*np.cos(dec), -np.sin(ra), -np.cos(ra)*np.sin(dec)],
             [np.sin(ra)*np.cos(dec), np.cos(ra), -np.sin(ra)*np.sin(dec)],
             [np.sin(dec), 0, np.cos(dec)]])
    return B

def get_jacobian(ra, dec, parallax, mu_ra, mu_dec, v_r):

    # Constants to improve readability
    c1 = k1/parallax
    c2 = -k1/(parallax**2)
    c3 = k2/parallax
    c4 = k2/(parallax**2)

    # deg -> radians
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)

    J11 = c1*(-np.sin(ra)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.cos(ra)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)))
    J12 = c1*(np.cos(dec)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN)) - np.sin(dec)*(np.cos(ra)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.sin(ra)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN))))
    J13 = c2*(np.cos(ra)*np.cos(dec)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.sin(ra)*np.cos(dec)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)) + np.sin(dec)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN)))
    J14 = 0
    J15 = 0
    J16 = 0

    J21 = c1*(-A[2,1]*np.sin(ra) + A[2,2]*np.cos(ra)) 
    J22 = c1*(-np.sin(dec)*(A[2,1]*np.cos(ra) + A[2,2]*np.sin(ra)) + A[2,3]*np.cos(dec))
    J23 = c2*(np.cos(dec)*(A[2,1]*np.cos(ra) + A[2,2]*np.sin(ra)) + A[2,3]*np.sin(dec))
    J24 = 0
    J25 = 0
    J26 = 0

    J31 = c1*(-np.sin(THETA_SUN)*(A[1,2]*np.cos(ra) - A[1,1]*np.sin(ra)) + np.cos(THETA_SUN)*(A[3,2]*np.cos(ra) - A[3,1]*np.sin(ra)))
    J32 = -c1*(np.sin(dec)*(np.cos(ra)*(A[1,1]*np.sin(THETA_SUN) - A[3,1]*np.cos(THETA_SUN)) + np.sin(ra)*(A[1,2]*np.sin(THETA_SUN) - A[3,2]*np.cos(THETA_SUN))) + np.cos(dec)*(A[3,3]*np.cos(THETA_SUN) - A[1,3]*np.sin(THETA_SUN)))
    J33 = c2*(np.cos(ra)*np.cos(dec)*(A[3,1]*np.cos(THETA_SUN) - A[1,1]*np.sin(THETA_SUN)) + np.sin(ra)*np.cos(dec)*(A[3,2]*np.cos(THETA_SUN) - A[1,2]*np.sin(THETA_SUN)) + np.sin(dec)*(A[3,3]*np.cos(THETA_SUN) - A[1,3]*np.sin(THETA_SUN)))
    J34 = 0
    J35 = 0
    J36 = 0

    J41 = (np.sin(ra)*(-np.cos(dec)*v_r + np.sin(dec)*c3*mu_dec) - np.cos(ra)*c3*mu_ra)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + (np.cos(ra)*(np.cos(dec)*v_r - np.sin(dec)*c3*mu_dec) - np.sin(ra)*c3*mu_ra)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN))
    J42 = np.cos(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.sin(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)) + (np.cos(dec)*v_r - np.sin(dec)*c3*mu_dec)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN))
    J43 = (np.sin(ra)*c4*mu_ra + np.cos(ra)*np.sin(dec)*c4*mu_dec)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + (-np.cos(ra)*c4*mu_ra + np.sin(ra)*np.sin(dec)*c4*mu_dec)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)) + (-np.cos(dec)*c4*mu_dec)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN))
    J44 = -np.sin(ra)*c3*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.cos(ra)*c3*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN))
    J45 = (-np.cos(ra)*np.sin(dec)*c3)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + (-np.sin(ra)*np.sin(dec)*c3)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)) + (np.cos(dec)*c3)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN))
    J46 = np.cos(ra)*np.cos(dec)*(A[1,1]*np.cos(THETA_SUN) + A[3,1]*np.sin(THETA_SUN)) + np.sin(ra)*np.cos(dec)*(A[1,2]*np.cos(THETA_SUN) + A[3,2]*np.sin(THETA_SUN)) + np.sin(dec)*(A[1,3]*np.cos(THETA_SUN) + A[3,3]*np.sin(THETA_SUN))

    J51 = A[2,1]*(-np.sin(ra)*np.cos(dec)*v_r - np.cos(ra)*c3*mu_ra + np.sin(ra)*np.sin(dec)*c3*mu_dec) + A[2,2]*(np.cos(ra)*np.cos(dec)*v_r - np.sin(ra)*c3*mu_ra - np.cos(ra)*np.sin(dec)*c3*mu_dec)
    J52 = A[2,1]*np.cos(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec) + A[2,2]*np.sin(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec) + A[2,3]*(np.cos(dec)*v_r - np.sin(dec)*c3*mu_dec)
    J53 = A[2,1]*(np.sin(ra)*c4*mu_ra + np.cos(ra)*np.sin(dec)*c4*mu_dec) + A[2,2]*(-np.cos(ra)*c4*mu_ra + np.sin(ra)*np.sin(dec)*c4*mu_dec) + A[2,3]*(-np.cos(dec)*c4*mu_dec)
    J54 = c3*(-A[2,1]*np.sin(ra) + A[2,2]*np.cos(ra))
    J55 = c3*(-A[2,1]*np.cos(ra)*np.sin(dec) - A[2,2]*np.sin(ra)*np.sin(dec) + A[2,3]*np.cos(dec))
    J56 = (A[2,1]*np.cos(ra) + A[2,2]*np.cos(ra))*np.cos(dec) + A[2,3]*np.sin(dec)

    J61 = (np.sin(ra)*(-np.cos(dec)*v_r + np.sin(dec)*c3*mu_dec) - np.cos(ra)*c3*mu_ra)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) + (np.cos(ra)*(np.cos(dec)*v_r - np.sin(dec)*c3*mu_dec) - np.sin(ra)*c3*mu_ra)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN))
    J62 = np.cos(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) + np.sin(ra)*(-np.sin(dec)*v_r - np.cos(dec)*c3*mu_dec)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN)) + (np.cos(dec)*v_r - np.sin(dec)*c3*mu_dec)*(A[3,3]*cos(THETA_SUN) - A[1,3]*np.sin*(THETA_SUN))
    J63 = c4*((np.sin(ra)*mu_ra + np.cos(ra)*np.sin(dec)*mu_dec)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) + (-np.cos(ra)*mu_ra + np.sin(ra)*np.sin(dec)*mu_dec)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN)) - np.cos(dec)*mu_dec*(A[3,3]*cos(THETA_SUN) - A[1,3]*np.sin*(THETA_SUN)))
    J64 = c3*(-np.sin(ra)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) + np.cos(ra)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN)))
    J65 = c3*(-np.cos(ra)*np.sin(dec)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) - np.sin(ra)*np.sin(dec)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN)) + np.cos(dec)*(A[3,3]*cos(THETA_SUN) - A[1,3]*np.sin*(THETA_SUN)))
    J66 = np.cos(ra)*np.cos(dec)*(A[3,1]*cos(THETA_SUN) - A[1,1]*np.sin*(THETA_SUN)) + np.sin(ra)*np.cos(dec)*(A[3,2]*cos(THETA_SUN) - A[1,2]*np.sin*(THETA_SUN)) + np.sin(dec)*(A[3,3]*cos(THETA_SUN) - A[1,3]*np.sin*(THETA_SUN))

    J = np.array([[J11, J12, J13, J14, J15, J16],
                  [J21, J22, J23, J24, J25, J26],
                  [J31, J32, J33, J24, J25, J26],
                  [J41, J42, J43, J44, J45, J46],
                  [J51, J52, J53, J54, J55, J56],
                  [J61, J62, J63, J64, J65, J66]])

    return J