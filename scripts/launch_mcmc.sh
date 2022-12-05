#!/bin/bash
set -e

python3 run_mcmc_asymmetric_h_fitting.py --cut-range 0.3 --nwalkers 50 --nsteps 10000 --nbins 10 --disk-scale 3000.0 --vlos-dispersion-scale 21000.0
#python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 9000 --nbins 10 --disk-scale 3000.0 --vlos-dispersion-scale 21000.0
# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 7000 --nbins 10 --disk-scale 4000.0 --vlos-dispersion-scale 16000.0

# # Vary disk scale length
# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 7000 --nbins 10 --disk-scale 2500.0 --vlos-dispersion-scale 16000.0
# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 7000 --nbins 10 --disk-scale 3000.0 --vlos-dispersion-scale 16000.0
# python3 run_mcmc_asymmetric_Vc_Only.py --cut-range 0.3 --nwalkers 50 --nsteps 7000 --nbins 10 --disk-scale 3500.0 --vlos-dispersion-scale 16000.0
