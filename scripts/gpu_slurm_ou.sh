#!/bin/bash
#SBATCH --job-name=ou_run
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:6
#SBATCH --mem-per-gpu=10G
#SBATCH -e /home/sven/repos/gaia-tools/out/ou_run_logs/error__%A.log
#SBATCH -o /home/sven/repos/gaia-tools/out/ou_run_logs/output__%A.log

IMG=/home/software/singularity/base.simg:2022-12-23.old
IMG=/home/software/singularity/base.simg:2023-11-07

# singularity exec -B /home -B /scratch -B /scratch-persistent --nv $IMG python3 scripts/gpu_test_script.py

singularity exec -B /local -B /home -B /scratch --nv $IMG \
    python3 scripts/run_mcmc_dr3_ou.py \
        --nwalkers 48 \
        --nsteps 6000 \
        --nbins 10 \
        --disk-scale 3000.0 \
        --vlos-dispersion-scale 21000.0 \
        --backend gpu
