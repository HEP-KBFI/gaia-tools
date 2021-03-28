# Gaia Tools


The purpose of this project is to develop code to help with Gaia data handling.

The long-term goal is to have a Gaia "toolbox" ready to use without having to reinvent the wheel a bunch of times.

[The gaia mission](https://sci.esa.int/web/gaia)  
<img src="https://sci.esa.int/documents/33580/35361/1567215149164-Gaia_mission_logo_625.jpg"  width="182" height="120">  

To start working on this project: 
1) Get the link to the repository by clicking on the blue "Clone" in the top right  
2) Copy the URL from e.g. the HTTPS field  
3) Open up an IDE of your choice and clone the repository by using the copied URL  
4) Once the repository is cloned to your project, you are ready to add code and update the GitLab project  

The point of using verion control is to make it easier for multiple people to work on the code base without submitting conflicting scripts.  
For each issue/task a separate code "branch" should be created. Once the task is complete, that same branch will be reviewed and merged into the master branch.  

## Requirements

The code is all written in python3 and makes substantial use of the standard pandas, numpy, matplotlib, etc. Additional libraries that you may need to investigate depending on your installation:

* [`astropy`](https://www.astropy.org/)
* [`mpl_scatter_density`](https://anaconda.org/conda-forge/mpl-scatter-density)
* [`emcee`](https://emcee.readthedocs.io/en/stable/)
* [`lmfit`](https://lmfit.github.io/lmfit-py/examples/example_brute.html)
* [`corner`](https://corner.readthedocs.io/en/latest/)
* [`h5py`](https://www.h5py.org/)


## Install package

To install the project as an importable package do the following:

1) Download as a tar.gz archive
2) Use pip install 'gaia_tools.tar.gz' or whatever the file name is
3) Inside code 'import gaia_tools' or other submodules inside the project
4) Done!

## Run in manivald

The following steps have to be taken in order to use the code in this repository in manivald:

- 1st step: Start singularity environment/image:
```bash
[manivald]$ singularity shell -B /hdfs -B /scratch /home/software/singularity/base_sven.simg
```
- 2nd step: Install `gaia-tools` (INSTALL THIS DIRECTLY IN SINGULARITY)
```bash
Singularity> export PYTHONPATH="/home/username/.local/lib/python3.6/site-packages/lib/python3.6/site-packages/"
Singularity> python3 setup.py install --prefix=/home/username/.local/lib/python3.6/site-packages
```
For more info on "singularity images", please visit [this github repository](https://github.com/HEP-KBFI/singularity).

### Jupyter notebooks
[This](https://ljvmiranda921.github.io/notebook/2018/01/31/running-a-jupyter-notebook/) is a fantastic guide on how to run a Jupyter notebook from a remote server.

### Data

Data is stored in `/scratch/mariacst/`  

To run the code properly you need to have in your data some specific parameters. From the Gaia archive these are represented under the following column names: 'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'. 'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr', 'radial_velocity', 'radial_velocity_error'


## Tutorials

The folder `jupyter-notebok` contains Jupyter Notebooks with examples of how to use this package. 
