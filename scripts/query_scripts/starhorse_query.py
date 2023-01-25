from pkg_resources import parse_version
import requests
import pyvo as vo
from pyvo.auth.authsession import AuthSession
import pandas as pd
import argparse
import time
from import_functions import import_data
#
# Verify the version of pyvo 
#
if parse_version(vo.__version__) < parse_version('1.0'):
    raise ImportError('pyvo version must larger than 1.0')
    
print('\npyvo version {version} \n'.format(version=vo.__version__))


'''
SELECT g."source_id", g."AV95", g."AG50", g."teff84", g."MG0"
FROM "gaiadr2_contrib"."starhorse" as "g", 
     "gaiadr2"."gaia_source" AS "s"
WHERE "s"."radial_velocity" IS NOT NULL 
AND "g"."source_id" = "s"."source_id"
LIMIT 10000000
'''


def run_single_job(args):

    outpath = args.out

    #
    # Setup tap_service
    #
    name = 'Gaia@AIP'
    url = "https://gaia.aip.de/tap"

    url_heidelberg = "http://dc.zah.uni-heidelberg.de/__system__/tap/run" 

    token = '9ef51a3b42860269f9cc7d5e2fa90cf026fc0815'

    print('TAP service %s \n' % name)

    # Change first line to needed parameters
    parameter_list = "g.source_id, g.AV95, g.AG50, g.teff84, g.MG0"

    starhorse_string = 'select top 50000 ' + parameter_list + \
    ' from gaiadr2_contrib.starhorse as g, gaiadr2.gaia_source as s ' \
    'where s.radial_velocity is not null ' \
    'and g.source_id = s.source_id'

    heidelberg_string = "select top all g.source_id from gdr2ap.main as g where g.source_id = {}"

    # Setup authorization
    tap_session = requests.Session()
    tap_session.headers['Authorization'] = token

    tap_service = vo.dal.TAPService(url, session=tap_session)

    tap_service = vo.dal.TAPService(url_heidelberg)
    print('Maxrec {}'.format(tap_service.maxrec))
    print('Hardlimit {}'.format(tap_service.hardlimit))

    my_path = "/hdfs/local/sven/gaia_tools_data/gaia_rv_data_bayes.csv"
    icrs_data = import_data(path = my_path, is_bayes = True, debug = True)

    for id in icrs_data.source_id:
        tap_result = tap_service.run_async(heidelberg_string, maxrec=10000000)
    
    
    tap_result.to_table()


    output_df = tap_result.to_table().to_pandas()
    output_df.to_csv(outpath)


def run_multiple_jobs(args):

    outpath = args.out
    #
    # Setup tap_service
    #
    name = 'Gaia@AIP'
    url = "https://gaia.aip.de/tap"
    token = '9ef51a3b42860269f9cc7d5e2fa90cf026fc0815'

    print('TAP service %s \n' % name)

    #
    # Setup authorisation
    #
    tap_session = requests.Session()
    tap_session.headers['Authorization'] = token

    tap_service = vo.dal.TAPService(url, session=tap_session)

    #
    # Submit queries
    #
    lang='PostgreSQL'
    jobs = []
    limit = 1000000
    total = 8000000
    
    parameter_list = "s.source_id,s.AV95,s.AG50,s.teff84,s.MG0"
    base_query = "select " + parameter_list + " from gaiadr2_contrib.starhorse as s, gaiadr2.gaia_source as g WHERE g.radial_velocity IS NOT NULL AND s.source_id = g.source_id LIMIT {limit:d} OFFSET {offset:d}"
    
    i=0
    for offset in range(0, total, limit):

        query = base_query.format(limit=limit, offset=offset)
        print(query)

        is_ok = False

        while is_ok == False:
            try:
                job = tap_service.submit_job(query, language=lang, runid='batch_'+str(i))
                job.run()
                jobs.append(job)
                
                is_ok = True
            except:
                print("An exception occurred")

        i = i + 1

    #
    # Collect the results
    #
    frames = ()
    for job in jobs:

        print('getting results from ' + str(job.job.runid))
        job.raise_if_error()

        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=10.)
        print(str(job.job.runid) + ' ' + str(job.phase))

        if job.phase in ("ERROR", "ABORTED"):
            pass

        else:
            tap_result = job.fetch_result()
            frames = frames + (tap_result.to_table().to_pandas(),)

    #
    # Contatenate into a pandas.DataFrame
    #
    df_results = pd.concat(frames)
    df_results.head()
    df_results.to_csv(outpath)


def query_AIP_tmass(args):

    outpath = args.out
    #
    # Setup tap_service
    #
    name = 'Gaia@AIP'
    url = "https://gaia.aip.de/tap"
    token = '9ef51a3b42860269f9cc7d5e2fa90cf026fc0815'
    print('TAP service %s \n' % name)

    #
    # Setup authorisation
    #
    tap_session = requests.Session()
    tap_session.headers['Authorization'] = token
    tap_service = vo.dal.TAPService(url, session=tap_session)

    #
    # Submit queries
    #
    lang='ADQL'
    
    # SELECT TOP 10 gaia.ra, gaia.dec,
    #     gaia.phot_g_mean_mag, gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag,
    #     tmass.j_m, tmass.h_m, tmass.k_m, tmass.ph_qual
    # FROM gaiadr2.gaia_source AS gaia,
    #     gaiadr2.tmass_best_neighbour AS xm,
    #     catalogs.tmass AS tmass
    # WHERE gaia.source_id = xm.source_id
    # AND xm.tmass_oid = tmass.tmass_oid;

    # THIS SHOULD WORK BUT IT DOESNT :(
    parameter_list = "rv.source_id, tm.tmass_oid, tm.j_m, tm.j_msigcom, " \
                    "tm.h_m, tm.h_msigcom, tm.k_m, tm.k_msigcom, tm.ph_qual"
    base_query = "select " + parameter_list + " from gaia_user_spoder.crossmatch_tmass_IDs_chunk_{} as rv, catalogs.tmass as tm " \
                "WHERE rv.tmass_oid = tm.tmass_oid"
    
    frames = ()

    for i in range(6):
        query = base_query.format(i)
        print(query)

        is_ok = False

        while is_ok == False:
            try:
                tap_result = tap_service.run_async(query, language=lang)
                tap_result.to_table()

                frames = frames + (tap_result.to_table().to_pandas(),)

                is_ok = True

            except Exception as e:
                print("An exception occurred: {}".format(e))
        time.sleep(1)
        i = i + 1

    #
    # Contatenate into a pandas.DataFrame
    #
    df_results = pd.concat(frames)
    df_results.to_csv(outpath)


if __name__ == "__main__":

    print('main')
    parser = argparse.ArgumentParser(description='Query StarHorse archive.')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    run_single_job(args)
    #query_AIP_tmass(args)