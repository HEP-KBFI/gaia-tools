# Error propagation

## Covariance data processing

## Covariance data propagation
Transforming the covariance matrix form ICRS to Galactocentric $v_x$, $v_y$, $v_z$.

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