#!/bin/bash
set -e

# python3 scripts/gaia_query.py --out /scratch/sven/gaia_downloads/bayesian_distance_rv_stars.csv \
#                 --login \
#                 --query-size 10000000

# python3 scripts/starhorse_query.py --out /scratch/sven/gaia_downloads/crossmatched_rv_tmass_data.csv


python3 scripts/gaia_query.py --out /scratch/sven/gaia_downloads/gaia_dr3_xm_2MASS_photometry_orig_ext_id.csv \
                                --login