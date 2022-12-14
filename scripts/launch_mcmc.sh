#!/bin/bash
set -e

python3 run_mcmc_z0_fitting.py --cut-range 0.3 --nwalkers 25 --nsteps 4000 --nbins 10 --disk-scale 3000.0 --vlos-dispersion-scale 21000.0

