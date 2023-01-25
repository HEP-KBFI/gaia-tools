from tabnanny import verbose
from xml.etree.ElementTree import TreeBuilder
from astroquery.gaia import Gaia
import warnings
import argparse
from pathlib import Path
# Comment this out if you want to see warnings
warnings.filterwarnings('ignore')

# Query string with most GAIA parameters
# ADQL_string = 'SELECT ALL gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec,gaia_source.dec_error, \
#     gaia_source.parallax,gaia_source.parallax_error,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error, \
#     gaia_source.ra_dec_corr,gaia_source.ra_parallax_corr,gaia_source.ra_pmra_corr,gaia_source.ra_pmdec_corr,gaia_source.dec_parallax_corr, \
#     gaia_source.dec_pmra_corr,gaia_source.dec_pmdec_corr,gaia_source.parallax_pmra_corr,gaia_source.parallax_pmdec_corr,gaia_source.pmra_pmdec_corr, \
#     gaia_source.ruwe,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_mag,gaia_source.radial_velocity, \
#     gaia_source.radial_velocity_error,gaia_source.phot_variable_flag, t.clean_tmass_psc_xsc_oid \
#     FROM gaiadr3.gaia_source as gaia_source, gaiadr3.tmass_psc_xsc_best_neighbour as t WHERE gaia_source.radial_velocity is not NULL and gaia_source.source_id = t.source_id'


#     # Select Bayesian r_estimates for stars with measured radial velocity
#     # ADQL_string = 'select all g.source_id, g.r_est from external.gaiadr2_geometric_distance as g, ' \
#     #                 'gaiadr2.gaia_source as s where s.radial_velocity is not null ' \
#     #                 'and g.source_id = s.source_id'



def launch_query(args):

    print("Start query.")
    if args.login:
        Gaia.login()

    ADQL_string_split_1 = 'SELECT ALL gaia_source.source_id,gaia_source.ra,gaia_source.ra_error,gaia_source.dec,gaia_source.dec_error, \
    gaia_source.parallax,gaia_source.parallax_error,gaia_source.pmra,gaia_source.pmra_error,gaia_source.pmdec,gaia_source.pmdec_error, \
    gaia_source.ra_dec_corr,gaia_source.ra_parallax_corr,gaia_source.ra_pmra_corr,gaia_source.ra_pmdec_corr,gaia_source.dec_parallax_corr, \
    FROM gaiadr3.gaia_source as gaia_source, gaiadr3.tmass_psc_xsc_best_neighbour as t WHERE gaia_source.radial_velocity is not NULL and gaia_source.source_id = t.source_id'

    ADQL_string_split_2 = 'SELECT ALL \
    gaia_source.ruwe,gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_mag,gaia_source.radial_velocity, \
    gaia_source.radial_velocity_error,gaia_source.phot_variable_flag, t.clean_tmass_psc_xsc_oid \
    FROM gaiadr3.gaia_source as gaia_source, gaiadr3.tmass_psc_xsc_best_neighbour as t WHERE gaia_source.radial_velocity is not NULL \
    and gaia_source.source_id = t.source_id'

    query_strings = [ADQL_string_split_1, ADQL_string_split_2]

    query = 'SELECT ALL gaia.source_id, tm.tmass_oid, tm.j_m, tm.j_msigcom, tm.h_m, tm.h_msigcom, tm.ks_m, tm.ks_msigcom, tm.ph_qual, tm.ra, tm.dec \
    FROM gaiadr3.gaia_source AS gaia \
    JOIN gaiaedr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id) \
    JOIN gaiaedr3.tmass_psc_xsc_join AS xjoin \
    ON xmatch.original_ext_source_id = xjoin.original_psc_source_id \
    JOIN gaiadr1.tmass_original_valid AS tm \
    ON xjoin.original_psc_source_id = tm.designation \
    WHERE gaia.radial_velocity is not NULL'

    outpath = args.out

    job = Gaia.launch_job_async(query, output_file=outpath, output_format='csv', dump_to_file=True, verbose=True, name='gaiadr3_xm_tmass_orig_ext_id')
    results = job.get_results()


    # else:
    #     for i, query in enumerate(query_strings):

    #         path = Path(outpath)
    #         path.with_name(path.stem + '_split{}'.format(i) + path.suffix)

    #         job = Gaia.launch_job_async(query, output_file=outpath, output_format='csv', dump_to_file=True, verbose=True)
    #         results = job.get_results()

    print("End query.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Query Gaia archive.')
    parser.add_argument('--out', type=str)
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--no-login', action='store_false')
    # parser.add_argument('--query-size', type=int)
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    launch_query(args)