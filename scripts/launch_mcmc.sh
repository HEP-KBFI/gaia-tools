#!/bin/bash
set -e

# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.15 --nwalkers 50 --nsteps 1500 --nbins 10

python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 1000 --nbins 5

# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.45 --nwalkers 50 --nsteps 1500 --nbins 10