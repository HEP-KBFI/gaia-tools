#!/bin/bash
set -e

python3 run_mcmc_dr3.py --nwalkers 48 \
                        --nsteps 10000 \
                        --nbins 10 \
                        --disk-scale 3000.0 \
                        --vlos-dispersion-scale 21000.0

