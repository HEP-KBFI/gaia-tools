#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 2
#SBATCH --mem-per-gpu=10G
#SBATCH -e /home/sven/repos/gaia-tools/out/logs/error__%A.log
#SBATCH -o /home/sven/repos/gaia-tools/out/logs/output__%A.log

IMG=/home/software/singularity/base.simg

# singularity exec -B /home -B /scratch -B /scratch-persistent --nv $IMG python3 scripts/gpu_test_script.py

singularity exec -B /local -B /home -B /scratch -B /scratch-persistent --nv $IMG \
    python3 scripts/mcmc_gpu_slurm.py \
        --nwalkers 48 \
        --nsteps 50000 \
        --nbins 10 \
        --disk-scale 3000.0 \
        --vlos-dispersion-scale 21000.0 \
        --backend gpus
