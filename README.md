# Gaia Tools

This repository contains code to help with general Gaia data handling and to perform a Bayesian analysis of the kinematics of stars in the Milky Way as described in [XXXX].


## Requirements

The code is all written in python3 and makes substantial use of the standard pandas, numpy, matplotlib, etc. Additional libraries that you may need to investigate depending on your installation:

* [`astropy`](https://www.astropy.org/)
* [`mpl_scatter_density`](https://anaconda.org/conda-forge/mpl-scatter-density)
* [`emcee`](https://emcee.readthedocs.io/en/stable/)
* [`lmfit`](https://lmfit.github.io/lmfit-py/examples/example_brute.html)
* [`corner`](https://corner.readthedocs.io/en/latest/)
* [`h5py`](https://www.h5py.org/)
* [`numba`](http://numba.pydata.org/)


## Install package

To install the project as an importable package do the following:

1) Download as a tar.gz archive
2) Use pip install 'gaia_tools.tar.gz' or whatever the file name is
3) Inside code 'import gaia_tools' or other submodules inside the project
4) Done!


## Data

The input data used in the paper XXXX can be found on Zenodo at: [XXXX].


## Tutorials

The folder `jupyter-notebok` contains Jupyter Notebooks with examples of how to use this package. 
