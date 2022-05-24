from astroquery.gaia import Gaia
import warnings
import argparse
# Comment this out if you want to see warnings
warnings.filterwarnings('ignore')

# Example query string
#ADQL_string = "select top 100 * from gaiadr2.gaia_source order by source_id"

def launch_query(args):

    if args.login:
        Gaia.login()

    # Select stars with measured radial velocity
    # ADQL_string = 'select top {} gaia_source.source_id,gaia_source.phot_g_mean_flux,gaia_source.phot_g_mean_flux_error,' \
    #                 'gaia_source.phot_g_mean_mag,gaia_source.phot_bp_mean_flux,gaia_source.phot_bp_mean_flux_error, '\
    #                 'gaia_source.phot_bp_mean_mag,gaia_source.phot_rp_mean_flux,gaia_source.phot_rp_mean_flux_error, '\
    #                 'gaia_source.phot_rp_mean_mag,gaia_source.bp_rp,gaia_source.bp_g,gaia_source.g_rp from'\
    #                 'gaiadr2.gaia_source where radial_velocity is not null'.format(args.query_size)


    # Select Bayesian r_estimates of stars with measured radial velocity
    ADQL_string = 'select all g.source_id, g.r_est from external.gaiadr2_geometric_distance as g, ' \
                    'gaiadr2.gaia_source as s where s.radial_velocity is not null ' \
                    'and g.source_id = s.source_id'

    outpath = args.out
    job = Gaia.launch_job_async(ADQL_string, output_file=outpath, output_format='csv', dump_to_file=True, verbose=True)

    print("End query.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Query Gaia archive.')
    parser.add_argument('--out', type=str)
    parser.add_argument('--login', action='store_true')
    parser.add_argument('--no-login', action='store_false')
    parser.add_argument('--query-size', type=int)
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    launch_query(args)