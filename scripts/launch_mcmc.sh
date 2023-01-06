#!/bin/bash
set -e

python3 run_mcmc_dr3_h_fitting.py --nwalkers 48 \
                        --nsteps 5000 \
                        --nbins 10 \
                        --disk-scale 3000.0 \
                        --vlos-dispersion-scale 21000.0

