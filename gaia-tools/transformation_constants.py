'''
File for storing different constants and requisite matrices for transforming coordinates to 
a different frame of reference.
'''

import numpy as np

#region CONSTANTS

# Distance to the galactic centre in pc
R_0 = 8178

# Distance above galactic midplane in pc
Z_0 = 27

# Angle theta (in rad)
THETA_0 = np.arcsin(Z_0/R_0)

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
H = np.array([[np.cos(THETA_0), 0, np.sin(THETA_0)],
             [0, 1, 0],
             [-np.sin(THETA_0), 0, np.cos(THETA_0)]])

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

def get_cylindrical_velocity_matrix(phi):
   
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    M = np.array([[cos_phi, sin_phi, 0],
             [-sin_phi, cos_phi, 0],
             [0, 0, 1]])
    return M


def get_jacobian(sub_df, coordinate_system):

    if(coordinate_system == "Cartesian"):

        # TODO: Implement exception handling!

        ra = sub_df.ra
        dec = sub_df.dec
        parallax = sub_df.parallax
        mu_ra = sub_df.pmra
        mu_dec = sub_df.pmdec
        v_r = sub_df.radial_velocity

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

 
        J11 = c1*(-sin_ra*(A[0,0]*cos_theta + A[2,0]*sin_theta) + cos_ra*(A[0,1]*cos_theta + A[2,1]*sin_theta))
        J12 = c1*(cos_dec*(A[0,2]*cos_theta + A[2,2]*sin_theta) - cos_dec*(cos_ra*(A[0,0]*cos_theta + A[2,0]*sin_theta) + sin_ra*(A[0,1]*cos_theta + A[2,1]*sin_theta)))
        J13 = c2*(cos_ra*cos_dec*(A[0,0]*cos_theta + A[2,0]*sin_theta) + sin_ra*cos_dec*(A[0,1]*cos_theta + A[2,1]*sin_theta) + cos_dec*(A[0,2]*cos_theta + A[2,2]*sin_theta))
        J14 = 0
        J15 = 0
        J16 = 0
    
        J21 = c1*(-A[1,0]*sin_ra + A[1,1]*cos_ra) 
        J22 = c1*(-cos_dec*(A[1,0]*cos_ra + A[1,1]*sin_ra) + A[1,2]*cos_dec)
        J23 = c2*(cos_dec*(A[1,0]*cos_ra + A[1,1]*sin_ra) + A[1,2]*cos_dec)
        J24 = 0
        J25 = 0
        J26 = 0

        J31 = c1*(-sin_theta*(A[0,1]*cos_ra - A[0,0]*sin_ra) + cos_theta*(A[2,1]*cos_ra - A[2,0]*sin_ra))
        J32 = -c1*(cos_dec*(cos_ra*(A[0,0]*sin_theta - A[2,0]*cos_theta) + sin_ra*(A[0,1]*sin_theta - A[2,1]*cos_theta)) + cos_dec*(A[2,2]*cos_theta - A[0,2]*sin_theta))
        J33 = c2*(cos_ra*cos_dec*(A[2,0]*cos_theta - A[0,0]*sin_theta) + sin_ra*cos_dec*(A[2,1]*cos_theta - A[0,1]*sin_theta) + cos_dec*(A[2,2]*cos_theta - A[0,2]*sin_theta))
        J34 = 0
        J35 = 0
        J36 = 0

        J41 = (sin_ra*(-cos_dec*v_r + cos_dec*c3*mu_dec) - cos_ra*c3*mu_ra)*(A[0,0]*cos_theta + A[2,0]*sin_theta) + (cos_ra*(cos_dec*v_r - cos_dec*c3*mu_dec) - sin_ra*c3*mu_ra)*(A[0,1]*cos_theta + A[2,1]*sin_theta)
        J42 = cos_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec)*(A[0,0]*cos_theta + A[2,0]*sin_theta) + sin_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec)*(A[0,1]*cos_theta + A[2,1]*sin_theta) + (cos_dec*v_r - cos_dec*c3*mu_dec)*(A[0,2]*cos_theta + A[2,2]*sin_theta)
        J43 = (sin_ra*c4*mu_ra + cos_ra*cos_dec*c4*mu_dec)*(A[0,0]*cos_theta + A[2,0]*sin_theta) + (-cos_ra*c4*mu_ra + sin_ra*cos_dec*c4*mu_dec)*(A[0,1]*cos_theta + A[2,1]*sin_theta) + (-cos_dec*c4*mu_dec)*(A[0,2]*cos_theta + A[2,2]*sin_theta)
        J44 = -sin_ra*c3*(A[0,0]*cos_theta + A[2,0]*sin_theta) + cos_ra*c3*(A[0,1]*cos_theta + A[2,1]*sin_theta)
        J45 = (-cos_ra*cos_dec*c3)*(A[0,0]*cos_theta + A[2,0]*sin_theta) + (-sin_ra*cos_dec*c3)*(A[0,1]*cos_theta + A[2,1]*sin_theta) + (cos_dec*c3)*(A[0,2]*cos_theta + A[2,2]*sin_theta)
        J46 = cos_ra*cos_dec*(A[0,0]*cos_theta + A[2,0]*sin_theta) + sin_ra*cos_dec*(A[0,1]*cos_theta + A[2,1]*sin_theta) + cos_dec*(A[0,2]*cos_theta + A[2,2]*sin_theta)

        J51 = A[1,0]*(-sin_ra*cos_dec*v_r - cos_ra*c3*mu_ra + sin_ra*cos_dec*c3*mu_dec) + A[1,1]*(cos_ra*cos_dec*v_r - sin_ra*c3*mu_ra - cos_ra*cos_dec*c3*mu_dec)
        J52 = A[1,0]*cos_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec) + A[1,1]*sin_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec) + A[1,2]*(cos_dec*v_r - cos_dec*c3*mu_dec)
        J53 = A[1,0]*(sin_ra*c4*mu_ra + cos_ra*cos_dec*c4*mu_dec) + A[1,1]*(-cos_ra*c4*mu_ra + sin_ra*cos_dec*c4*mu_dec) + A[1,2]*(-cos_dec*c4*mu_dec)
        J54 = c3*(-A[1,0]*sin_ra + A[1,1]*cos_ra)
        J55 = c3*(-A[1,0]*cos_ra*cos_dec - A[1,1]*sin_ra*cos_dec + A[1,2]*cos_dec)
        J56 = (A[1,0]*cos_ra + A[1,1]*cos_ra)*cos_dec + A[1,2]*cos_dec

        J61 = (sin_ra*(-cos_dec*v_r + cos_dec*c3*mu_dec) - cos_ra*c3*mu_ra)*(A[2,0]*cos_theta - A[0,0]*sin_theta) + (cos_ra*(cos_dec*v_r - cos_dec*c3*mu_dec) - sin_ra*c3*mu_ra)*(A[2,1]*cos_theta - A[0,1]*sin_theta)
        J62 = cos_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec)*(A[2,0]*cos_theta - A[0,0]*sin_theta) + sin_ra*(-cos_dec*v_r - cos_dec*c3*mu_dec)*(A[2,1]*cos_theta - A[0,1]*sin_theta) + (cos_dec*v_r - cos_dec*c3*mu_dec)*(A[2,2]*cos_theta - A[0,2]*sin_theta)
        J63 = c4*((sin_ra*mu_ra + cos_ra*cos_dec*mu_dec)*(A[2,0]*cos_theta - A[0,0]*sin_theta) + (-cos_ra*mu_ra + sin_ra*cos_dec*mu_dec)*(A[2,1]*cos_theta - A[0,1]*sin_theta) - cos_dec*mu_dec*(A[2,2]*cos_theta - A[0,2]*sin_theta))
        J64 = c3*(-sin_ra*(A[2,0]*cos_theta - A[0,0]*sin_theta) + cos_ra*(A[2,1]*cos_theta - A[0,1]*sin_theta))
        J65 = c3*(-cos_ra*cos_dec*(A[2,0]*cos_theta - A[0,0]*sin_theta) - sin_ra*cos_dec*(A[2,1]*cos_theta - A[0,1]*sin_theta) + cos_dec*(A[2,2]*cos_theta - A[0,2]*sin_theta))
        J66 = cos_ra*cos_dec*(A[2,0]*cos_theta - A[0,0]*sin_theta) + sin_ra*cos_dec*(A[2,1]*cos_theta - A[0,1]*sin_theta) + cos_dec*(A[2,2]*cos_theta - A[0,2]*sin_theta)

        J = np.array([[J11, J12, J13, J14, J15, J16],
                        [J21, J22, J23, J24, J25, J26],
                        [J31, J32, J33, J24, J25, J26],
                        [J41, J42, J43, J44, J45, J46],
                        [J51, J52, J53, J54, J55, J56],
                        [J61, J62, J63, J64, J65, J66]])

    elif(coordinate_system == "Cylindrical"):

        # TODO: Implement exception handling!

        x = sub_df.x
        y = sub_df.y
        r = sub_df.r
        phi = sub_df.phi
        v_r = sub_df.v_r
        v_phi = sub_df.v_phi

        c1 = x/(r**2)
        c2 = y/(r**2)

        J11 = x/r
        J12 = y/r
        J13 = 0
        J14 = 0
        J15 = 0
        J16 = 0
    
        J21 = -c2 
        J22 = c1
        J23 = 0
        J24 = 0
        J25 = 0
        J26 = 0

        J31 = 0
        J32 = 0
        J33 = 1
        J34 = 0
        J35 = 0
        J36 = 0

        J41 = -v_phi*c2
        J42 = v_phi*c1
        J43 = 0 
        J44 = np.cos(phi)
        J45 = np.sin(phi)
        J46 = 0

        J51 = v_r*c2
        J52 = -v_r*c1
        J53 = 0 
        J54 = -np.sin(phi)
        J55 = np.cos(phi)
        J56 = 0

        J61 = 0
        J62 = 0
        J63 = 0
        J64 = 0
        J65 = 0
        J66 = 1
        
        J = np.array([[J11, J12, J13, J14, J15, J16],
                        [J21, J22, J23, J24, J25, J26],
                        [J31, J32, J33, J24, J25, J26],
                        [J41, J42, J43, J44, J45, J46],
                        [J51, J52, J53, J54, J55, J56],
                        [J61, J62, J63, J64, J65, J66]])

    return J
