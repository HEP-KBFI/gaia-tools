#!/bin/bash
set -e

python3 mcmc_gpu_test.py --nwalkers 25 \
                        --nsteps 15000 \
                        --nbins 5 \
                        --disk-scale 3000.0 \
                        --vlos-dispersion-scale 21000.0

