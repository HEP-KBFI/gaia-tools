#!/bin/bash
set -e

python3 run_mcmc_dr3.py --nwalkers 25 --nsteps 15000 --nbins 3 --disk-scale 3000.0 --vlos-dispersion-scale 21000.0 --backend gpu

