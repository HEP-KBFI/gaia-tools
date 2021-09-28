'''
File for storing different constants and requisite matrices for transforming coordinates to 
a different frame of reference.
'''

import numpy as np
from numba import njit, jit

#region CONSTANTS

# Distance to the galactic centre in pc
R_0 = 8178

# Distance above galactic midplane in pc
Z_0 = 17

# Angle theta (in rad)
# Now calculated only in function get_H_matrix()

#THETA_0 = np.arcsin(Z_0/R_0)

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

# Constant used for fixing the order of magnitude of distance
k1 = 10**3

# Constant for converting 1/yr to km/s
k2 = 4.74047

#endregion


# Matrix H which accounts for the height of the Sun (Z_0) aboce the Galactic plane.
def get_H_matrix(Z_0, R_0):
    
    if(Z_0/R_0 is None):
        print("Something went wrong! No values for either Z_0 or R_0 were found!")
        return
   
    THETA_0 = np.arcsin(Z_0/R_0)

    # Defining constants to reduce process time
    costheta = np.cos(THETA_0)
    sintheta = np.sin(THETA_0)

    H = np.array([[costheta, 0, sintheta],
                 [0, 1, 0],
                 [-sintheta, 0, costheta]])

    return H

'''
Get B matric for velocity transformations.

Expects ra, dec as NumPy arrays.
'''
@jit(nopython=True)
def get_b_matrix(ra, dec):

    #B = np.array([[cosra*cosdec, -sinra, -cosra*sindec],
    #         [sinra*cosdec, cosra, -sinra*sindec],
    #         [sindec, np.zeros(n), cosdec]])
    
    
    # Add check if ra == dec
    
    n = len(ra)

    B = np.zeros((n, 3, 3))
    
    # Defining constants to reduce process time
    cosra = np.cos(ra)
    cosdec = np.cos(dec)
    sinra = np.sin(ra)
    sindec = np.sin(dec)

    B[:, 0, 0] = cosra*cosdec
    B[:, 0, 1] = -sinra
    B[:, 0, 2] = -cosra*sindec
    
    B[:, 1, 0] = sinra*cosdec
    B[:, 1, 1] = cosra
    B[:, 1, 2] = -sinra*sindec
    
    B[:, 2, 0] = sindec
    B[:, 2, 2] = cosdec
    
    # Returns array of B matrices - one for each data point
    return B

def get_cylindrical_velocity_matrix(phi):
   
    n = len(phi)

    sin_phi = np.sin(phi).ravel()
    cos_phi = np.cos(phi).ravel()

    M = np.array([[cos_phi, sin_phi, np.zeros(n)],
         [-sin_phi, cos_phi, np.zeros(n)],
         [np.zeros(n), np.zeros(n), np.ones(n)]])

    M = M.T.reshape(n,3,3, order = 'A').swapaxes(1,2)

    # Returns array of matrices - one for each data point
    return M


@jit(nopython=True)
def get_jacobian(df, coordinate_system, Z_0, R_0):

    n = len(df)
    
    if(coordinate_system == "Cartesian"):

        # DF -> ["ra", "dec","parallax","pmra","pmdec","radial_velocity"]
        
        if(Z_0/R_0 is None):
            print("Something went wrong! No values for either Z_0 or R_0 were found!")
            return
   
        THETA_0 = np.arcsin(Z_0/R_0)

        # TODO: Implement exception handling!

        ra = df[:,0]
        dec = df[:,1]
        parallax = df[:,2]
        mu_ra = df[:,3]
        mu_dec = df[:,4]
        v_r = df[:,5]

        # Constants to improve readability
        c1 = k1/parallax
        c2 = -k1/(parallax**2)
        c3 = k2/parallax
        c4 = k2/(parallax**2)

        # deg -> radians
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)

        # Declaring variables to reduce number of computations 
        sin_ra = np.sin(ra)
        cos_ra = np.cos(ra)
        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)
        sin_theta = np.sin(THETA_0)
        cos_theta = np.cos(THETA_0)
        
        A_1 = A[0,0]*cos_theta + A[2,0]*sin_theta
        A_2 = A[0,1]*cos_theta + A[2,1]*sin_theta
        A_3 = A[0,2]*cos_theta + A[2,2]*sin_theta
        A_4 = -A[1,0]*sin_ra + A[1,1]*cos_ra
        A_5 = A[1,0]*cos_ra + A[1,1]*sin_ra
        A_6 = A[2,2]*cos_theta - A[0,2]*sin_theta
        A_7 = A[2,0]*cos_theta - A[0,0]*sin_theta
        A_8 = A[2,1]*cos_theta - A[0,1]*sin_theta
        
        cosra_cosdec = cos_ra*cos_dec
        cosra_sindec = cos_ra*sin_dec
        sinra_cosdec = sin_ra*cos_dec
        sinra_sindec = sin_ra*sin_dec

        expr_1 = cos_dec*v_r - sin_dec*c3*mu_dec
        expr_2 = -cos_dec*v_r + sin_dec*c3*mu_dec
        expr_3 = -sin_dec*v_r - cos_dec*c3*mu_dec
        expr_4 = sin_ra*c4*mu_ra + cosra_sindec*c4*mu_dec
        expr_5 = cos_ra*c3*mu_ra
        expr_6 = sin_ra*c3*mu_ra
        expr_7 = -cos_ra*c4*mu_ra + sinra_sindec*c4*mu_dec
        
        J11 = c1*(-sin_ra*(A_1) + cos_ra*(A_2))
        J12 = c1*(cos_dec*(A_3) - sin_dec*(cos_ra*(A_1) + sin_ra*(A_2)))
        J13 = c2*(cosra_cosdec*(A_1) + sinra_cosdec*(A_2) + sin_dec*(A_3))
        J14 = np.zeros(n)
        #J15 = np.zeros(n)
        #J16 = np.zeros(n)
        
        #row_1 = np.array([J11, J12, J13, J14, J15, J16])
        
        J21 = c1*(A_4) 
        J22 = c1*(-sin_dec*A_5 + A[1,2]*cos_dec) 
        J23 = c2*(cos_dec*A_5 + A[1,2]*sin_dec)
        # J24 = np.zeros(n)
        # J25 = np.zeros(n)
        # J26 = np.zeros(n)

        J31 = c1*(-sin_theta*(A[0,1]*cos_ra - A[0,0]*sin_ra) + cos_theta*(A[2,1]*cos_ra - A[2,0]*sin_ra))
        J32 = -c1*(sin_dec*(cos_ra*(A[0,0]*sin_theta - A[2,0]*cos_theta) + sin_ra*(A[0,1]*sin_theta - A[2,1]*cos_theta)) + cos_dec*(A_6))
        J33 = c2*(cosra_cosdec*(A_7) + sinra_cosdec*(A_8) + sin_dec*(A_6))
        # J34 = np.zeros(n)
        # J35 = np.zeros(n)
        # J36 = np.zeros(n)

        J41 = (sin_ra*(expr_2) - expr_5)*(A_1) + (cos_ra*(expr_1) - expr_6)*(A_2)
        J42 = cos_ra*(expr_3)*(A_1) + sin_ra*(expr_3)*(A_2) + (expr_1)*(A_3)
        J43 = (expr_4)*(A_1) + (expr_7)*(A_2) + (-cos_dec*c4*mu_dec)*(A_3)
        J44 = -sin_ra*c3*(A_1) + cos_ra*c3*(A_2)
        J45 = (-cosra_sindec*c3)*(A_1) + (-sinra_sindec*c3)*(A_2) + (cos_dec*c3)*(A_3)
        J46 = cosra_cosdec*(A_1) + sinra_cosdec*(A_2) + sin_dec*(A_3)

        J51 = A[1,0]*(-sinra_cosdec*v_r - expr_5 + sinra_sindec*c3*mu_dec) + A[1,1]*(cosra_cosdec*v_r - expr_6- cosra_sindec*c3*mu_dec)
        J52 = A[1,0]*cos_ra*(expr_3) + A[1,1]*sin_ra*(expr_3) + A[1,2]*(expr_1)
        J53 = A[1,0]*(expr_4) + A[1,1]*(expr_7) + A[1,2]*(-cos_dec*c4*mu_dec)
        J54 = c3*(A_4)
        J55 = c3*(-A[1,0]*cosra_sindec- A[1,1]*sinra_sindec + A[1,2]*cos_dec)
        J56 = (A[1,0]*cos_ra + A[1,1]*sin_ra)*cos_dec + A[1,2]*sin_dec

        J61 = (sin_ra*(expr_2) - expr_5)*(A_7) + (cos_ra*(expr_1) - expr_6)*(A_8)
        J62 = cos_ra*(expr_3)*(A_7) + sin_ra*(expr_3)*(A_8) + (expr_1)*(A_6)
        J63 = c4*((sin_ra*mu_ra + cosra_sindec*mu_dec)*(A_7) + (-cos_ra*mu_ra + sinra_sindec*mu_dec)*(A_8) - cos_dec*mu_dec*(A_6))
        J64 = c3*(-sin_ra*(A_7) + cos_ra*(A_8))
        J65 = c3*(-cosra_sindec*(A_7) - sinra_sindec*(A_8) + cos_dec*(A_6))
        J66 = cosra_cosdec*(A_7) + sinra_cosdec*(A_8) + sin_dec*(A_6)
        
        
        J1 = np.stack((J11, J12, J13, J14, J14, J14))
        J2 = np.stack((J21, J22, J23, J14, J14, J14))
        J3 = np.stack((J31, J32, J33, J14, J14, J14))
        J4 = np.stack((J41, J42, J43, J44, J45, J46))
        J5 = np.stack((J51, J52, J53, J54, J55, J56))
        J6 = np.stack((J61, J62, J63, J64, J65, J66))
        
        J = np.stack((J1, J2, J3, J4, J5, J6))

    elif(coordinate_system == "Cylindrical"):

        # TODO: Implement exception handling!

        # DF -> ["x", "y","r","phi","v_r","v_phi"]

        x = df[:,0]
        y = df[:,1]
        r = df[:,2]
        phi = df[:,3]
        v_r = df[:,4]
        v_phi = df[:,5]

    
        c1 = x/(r**2)
        c2 = y/(r**2)
        
        # Declaring variables to reduce number of computations 
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        J11 = x/r
        J12 = y/r
        J13 = np.zeros(n)
        #J14 = np.zeros(n)
        #J15 = np.zeros(n)
        #J16 = np.zeros(n)
    
        J21 = -c2 
        J22 = c1
        #J23 = np.zeros(n)
        #J24 = np.zeros(n)
        #J25 = np.zeros(n)
        #J26 = np.zeros(n)

        #J31 = np.zeros(n)
        #J32 = np.zeros(n)
        J33 = np.ones(n)
        #J34 = np.zeros(n)
        #J35 = np.zeros(n)
        #J36 = np.zeros(n)

        J41 = -v_phi*c2
        J42 = v_phi*c1
        #J43 = np.zeros(n)
        J44 = cos_phi
        J45 = sin_phi
        #J46 = np.zeros(n)

        J51 = v_r*c2
        J52 = -v_r*c1
        #J53 = np.zeros(n) 
        J54 = -sin_phi
        J55 = cos_phi
        #J56 = np.zeros(n)

        #J61 = np.zeros(n)
        #J62 = np.zeros(n)
        #J63 = np.zeros(n)
        #J64 = np.zeros(n)
        #J65 = np.zeros(n)
        #J66 = np.ones(n)
               
        J1 = np.stack((J11, J12, J13, J13, J13, J13))
        J2 = np.stack((J21, J22, J13, J13, J13, J13))
        J3 = np.stack((J13, J13, J33, J13, J13, J13))
        J4 = np.stack((J41, J42, J13, J44, J45, J13))
        J5 = np.stack((J51, J52, J13, J54, J55, J13))
        J6 = np.stack((J13, J13, J13, J13, J13, J33))
        
        J = np.stack((J1, J2, J3, J4, J5, J6))
        
        
    return J
