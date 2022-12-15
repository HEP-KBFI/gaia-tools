#!/bin/bash
set -e

python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 5000 --nbins 10 --disk-scale 3000.0 --vlos-dispersion-scale 21000.0

