
## Covariance data processing
A covariance matrix is a 6x6 matrix in the celestial frame from the dataframe of Gaia data. The parameter's columns used in this Pandas dataframe are: 'ra', 'dec', 'parallax, 'pmra', 'pmdec', and 'radial_velocity'. You should also have all the columns ending with '_error' for each of the parameters named before and columns ending with '_corr' for pairs that are correlated.

In the Gaia-Tools code base under 'covmatrices_generation' the 'generate_covmat' function is used to create the covariance matrices and the function 'generate_covmatrices' can be used to transform the matrices from one coordinate system to another.

## Covariance data propagation
Transforming the covariance matrix from ICRS to Galactocentric.

Here's an example:
```py
data_icrs=pd.read_csv(path)
from covariance_generation import generate_covmatrices, generate_covmat
from data_analysis import get_transformed_data
df_new=get_transformed_data(data_icrs, z_0=25, r_0=8277, is_bayes=False)
generate_covmatrices(data_icrs,
                         df_crt = df_new,
                         transform_to_galcen = True,
                         transform_to_cylindrical = False,
                         z_0 = 25,
                         r_0 = 8277,
                         is_bayes = False,
                         is_unpack_velocity = False,
                         debug = False)
```

Now we can transform the covariance matrix from Galactocentric $v_x$, $v_y$, $v_z$ to Galactocentric $v_r$, $v_\varphi$, $v_z$.

For example:
```py
data_icrs=pd.read_csv(path)
from covariance_generation import generate_covmatrices, generate_covmat
from data_analysis import get_transformed_data
df_new=get_transformed_data(data_icrs, z_0=25, r_0=8277, include_cylindrical=True, is_bayes=False)
generate_covmatrices(data_icrs,
                         df_crt = df_new,
                         transform_to_galcen = True,
                         transform_to_cylindrical = True,
                         z_0 = 25,
                         r_0 = 8277,
                         is_bayes = False,
                         is_unpack_velocity = False,
                         debug = False)
```

![Errorpropagation](figures/errorpropagation.png)

![Errorpropagation2](figures/err_prop2.png)
