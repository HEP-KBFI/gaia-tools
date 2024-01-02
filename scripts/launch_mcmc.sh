#!/bin/bash
set -e

IMG=/home/software/singularity/base.simg:2022-12-23.old

# singularity exec -B /home -B /scratch -B /scratch-persistent --nv $IMG python3 scripts/gpu_test_script.py

# singularity exec -B /local -B /home -B /scratch --nv $IMG \
#     python3 run_mcmc_dr3_h_fitting.py 
#                         --nwalkers 48 \
#                         --nsteps 8000 \
#                         --nbins 10 \
#                         --disk-scale 3000.0 \
#                         --vlos-dispersion-scale 21000.0

# singularity exec -B /local -B /home -B /scratch --nv $IMG \
#     python3 run_mcmc_dr3_vc_only.py \
#                         --nwalkers 48 \
#                         --nsteps 8000 \
#                         --nbins 9 \
#                         --disk-scale 3000.0 \
#                         --vlos-dispersion-scale 21000.0


singularity exec -B /local -B /home -B /scratch --nv $IMG \
    python3 filtered_DR3_dataset.py

