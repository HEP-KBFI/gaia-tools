#!/bin/bash
set -e

python3 scripts/gaia_query.py --out /scratch/sven/gaia_downloads/photometry_with_rv.csv \
                --login \
                --query-size 10000000