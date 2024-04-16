# Galactocentric transformation
These codes require six parameters from Gaia data: right ascension $\alpha$ [deg], declination $\delta$ [deg], parallax $\varpi$ [mas] and their velocities $\mu_\alpha$ [mas/year], $\mu_\delta$ [mas/year], $v_r$ [km/s]. 
We want to transform these coordinates to Galactrocentric Cartesian coordinates: x, y, z, $v_x$, $v_y$, $v_z$ and then to Galactocentric Cylindrical coordinates: r, $\varphi$, z, $v_r$, $v_\varphi$, $v_z$.

```mermaid
graph TD
    A[ICRS] --> B[Galactocentric Cartesian];
    B -->C[Galactocentric Cylindrical];
```

All Gaia parameters can be found [here.](https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html) 
## Positional observables
Transforming ICRS coordinates to Galactocentric x, y, z.

Here's an example:
```py
from data_analysis import get_transformed_data
df=pd.read_csv(path)
df_new=get_transformed_data(df,
                        include_cylindrical = False,
                        z_0 = 17,
                        r_0 = 8178,
                        v_sun = ([[11.1], [232.24], [7.25]])
                        is_bayes = False,
                        is_output_frame = False,
                        is_source_included = False,
                        NUMPY_LIB = np,
                        dtype = np.float64)
```
ICRS:

![ICRS](figures/ICRS.png)

<!---<img src='figures/ICRS.png' alt='ICRS' title='ICRS' width='100' height='100'/>-->

Galactocentric Cartesian:

![Cartesian](figures/Cartesian.png)

test