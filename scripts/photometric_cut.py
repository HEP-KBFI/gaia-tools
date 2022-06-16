
import re
import vaex
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np

# Fouesneau Columns
columns = ['source_id', 'J_p50', 'Ks_p50', 'H_p50', 'J_sigma', 'Ks_sigma', 'H_sigma', 'parallax', 'parallax_sigma']


def get_sample_IDs(start_datetime, is_save_region):

    # Import Fouesneau
    vaex_path = '/scratch/sven/gaia_downloads/catalog-20210311-goodphot_lite_nodup.vaex.hdf5'

    df=vaex.open(vaex_path)
    df['source_id'] = df.source_id.astype('int64')
    df = df[columns]

    # Import 2MASS data to data frame
    tmass_data_path = '/scratch/sven/gaia_downloads/crossmatched_tmass_data.csv'
    crossmatched_tmass_data = vaex.from_csv(tmass_data_path, convert=True)

    crossmatched_tmass_data = crossmatched_tmass_data.join(df, how='inner', 
                                left_on='source_id', 
                                right_on='source_id',
                                lsuffix='_tmass_xmass',
                                rsuffix='_fouesnau')

    # This next import is probably redundant
    gaia_rv_data = vaex.from_csv('/scratch/sven/gaia_downloads/bayesian_distance_rv_stars.csv', convert=True)

    crossmatched_tmass_data = crossmatched_tmass_data.join(gaia_rv_data, how='inner', 
                                left_on='source_id_tmass_xmass', 
                                right_on='source_id',
                                lsuffix='_tmass_xmass',
                                rsuffix='_rv_data')

    print("Stars in the sample before making photometric cuts: {}".format(crossmatched_tmass_data.shape))

    # To make sure all photometric quality flags are "A" or "B"
    def is_allowed_flg(string):

        charRe = re.compile(r'[^A-B]')
        string = charRe.search(string)
        return not bool(string)

    crossmatched_tmass_data['is_qual_true'] = crossmatched_tmass_data.ph_qual.apply(is_allowed_flg)

    cut_sample = crossmatched_tmass_data[((crossmatched_tmass_data.J_p50 - crossmatched_tmass_data.Ks_p50) > 0.5) & ((crossmatched_tmass_data.J_p50 - crossmatched_tmass_data.Ks_p50) < 1.1)]
    cut_sample = cut_sample[(cut_sample.J_sigma < 0.1) & (cut_sample.Ks_sigma < 0.1) & (cut_sample.H_sigma < 0.1)]
    cut_sample =  cut_sample[cut_sample.is_qual_true == True]
    cut_sample = cut_sample[cut_sample.parallax_sigma/cut_sample.parallax < 0.2]

    print("Stars in the sample after making photometric cuts: {}".format(cut_sample.shape))

    #
    # Bin data on 2D histogram and save final selected region
    # 

    xlim_min = 0.4
    xlim_max = 1.2
    ylim_min = -3
    ylim_max = 0

    fig = plt.figure(figsize=(8, 8))

    range = [[xlim_min, xlim_max], [ylim_min, ylim_max]]
    x = (cut_sample.J_p50 - cut_sample.Ks_p50).values
    y = cut_sample.H_p50.values
    h = plt.hist2d(x, y, bins=250, range=range, cmin=50, norm=colors.PowerNorm(0.5), zorder=0.5)
    plt.scatter(x, y, alpha=0.05, s=1, color='k', zorder=0)

    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    plt.colorbar(h[3], pad=0.02, format=fmt, orientation='vertical')

    thresh_max = 200
    xc, yc = np.where(h[0] > thresh_max)
    fit_x = h[1][xc]
    fit_y = h[2][yc]
    cut_ids = np.where((fit_y > -2) & (fit_y < -1))
    a, b = np.polyfit(fit_x[cut_ids], fit_y[cut_ids], 1)
    plt.plot(h[1], a*h[1] + b, c='w')

    plt.plot(h[1], a*h[1] + b + 0.3, c='w', linestyle="--")
    plt.plot(h[1], a*h[1] + b - 0.3, c='w', linestyle="--")

    plt.grid()
    
    if(is_save_region):
        fig_name = 'cut_region_' + start_datetime
        plt.savefig('/home/sven/repos/gaia-tools/out/mcmc_plots/photometric_cut/' + fig_name +'.png', dpi=300, bbox_inches='tight', facecolor='white')

    #
    # Impose CUT for RC region
    #

    JKs = cut_sample.J_p50 - cut_sample.Ks_p50
    cut_sample = cut_sample[(cut_sample.H_p50 < a*JKs + b + 0.3) & (cut_sample.H_p50 > a*JKs + b - 0.3)]

    # Save final sample IDs to pandas DataFrame
    sample_IDs = cut_sample.to_pandas_df(['source_id', 'r_est'])

    return sample_IDs