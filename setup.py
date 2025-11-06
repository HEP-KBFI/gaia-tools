import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaia_tools",
    version="0.1.4",
    author="Sven Poder",
    author_email="sven.poder@live.com",
    description="The 'gaia-tools' package provides utilities for working with Gaia astrometric data. It includes functions for transforming coordinates between ICRS and Galactocentric (Cartesian and cylindrical) frames, as well as tools for constructing and propagating full astrometric covariance matrices. The package is designed to support analyses of Galactic dynamics.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.hep.kbfi.ee/gaia-physics/gaia-tools",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "numba"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
