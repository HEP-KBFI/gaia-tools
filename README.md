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
