
import re
import vaex
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import numpy as np

# Fouesneau Columns
columns = ['source_id', 'J_p50', 'Ks_p50', 'H_p50', 'J_sigma', 'Ks_sigma', 'H_sigma', 'parallax', 'parallax_sigma']


def get_sample_IDs(run_out_path, cut_range, is_save_region=True):

    # Import Fouesneau
    vaex_path = '/scratch/sven/gaia_tools_data/extinction_corrected_photometry/catalog-20210311-goodphot_lite_nodup.vaex.hdf5'

    df=vaex.open(vaex_path)
    df['source_id'] = df.source_id.astype('int64')
    df = df[columns]

    # Import 2MASS data to data frame
    tmass_data_path = '/home/svenpoder/DATA/Gaia_2MASS Data_DR2/gaia_tools_data/crossmatched_tmass_data/crossmatched_tmass_data.csv'
    crossmatched_tmass_data = vaex.from_csv(tmass_data_path, convert=True)

    crossmatched_tmass_data = crossmatched_tmass_data.join(df, how='inner', 
                                left_on='source_id', 
                                right_on='source_id',
                                lsuffix='_tmass_xmass',
                                rsuffix='_fouesnau')

    # This next import is probably redundant
    gaia_rv_data = vaex.from_csv('/home/svenpoder/DATA/Gaia_2MASS Data_DR2/gaia_tools_data/bayesian_distance_rv_stars.csv', convert=True)

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
    #cut_sample = crossmatched_tmass_data[((crossmatched_tmass_data.J_p50 - crossmatched_tmass_data.Ks_p50) > 0.6) & ((crossmatched_tmass_data.J_p50 - crossmatched_tmass_data.Ks_p50) < 0.9)]


    if(is_save_region):
        # Plot sample before uncertainty cut
        plot_filter_uncertainties(cut_sample, run_out_path, 'Sample before the cut JHKS_sigma < 0.1')
    cut_sample = cut_sample[(cut_sample.J_sigma < 0.1) & (cut_sample.Ks_sigma < 0.1) & (cut_sample.H_sigma < 0.1)]

    if(is_save_region):
        # Plot sample before photometric quality cut
        plot_filter_uncertainties_qual(cut_sample, run_out_path, is_save = True)
    cut_sample =  cut_sample[cut_sample.is_qual_true == True]

    if(is_save_region):
        # Plot sample before parallax sigma cut
        plot_filter_uncertainties_w_parallax(cut_sample, run_out_path)
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
    plt.colorbar(h[3], pad=0.02, format=fmt, orientation='vertical', label = 'Star density')

    thresh_max = 200
    xc, yc = np.where(h[0] > thresh_max)
    fit_x = h[1][xc]
    fit_y = h[2][yc]
    cut_ids = np.where((fit_y > -2) & (fit_y < -1))
    a, b = np.polyfit(fit_x[cut_ids], fit_y[cut_ids], 1)
    
    if(is_save_region):
        plt.plot(h[1], a*h[1] + b, c='w')
        plt.plot(h[1], a*h[1] + b + cut_range, c='w', linestyle="--")
        plt.plot(h[1], a*h[1] + b - cut_range, c='w', linestyle="--")
        plt.xlabel("J - Ks [mag]", fontdict={'fontsize': 15})
        plt.ylabel("H [mag]", fontdict={'fontsize': 15})
        plt.grid()

        plt.xlim(0.5, 1.1)
        plt.ylim(-3.0, 0)
    
        fig_name = '/cut_region'

        plt.savefig(run_out_path + fig_name +'.png', dpi=300, bbox_inches='tight', facecolor='white')

    #
    # Impose CUT for RC region
    #

    JKs = cut_sample.J_p50 - cut_sample.Ks_p50
    cut_sample = cut_sample[(cut_sample.H_p50 < a*JKs + b + cut_range) & (cut_sample.H_p50 > a*JKs + b - cut_range)]

    # Save final sample IDs to pandas DataFrame
    sample_IDs = cut_sample.to_pandas_df(['source_id', 'r_est'])

    return sample_IDs

def plot_filter_uncertainties(cut_sample, out, suptitle='', is_save = True):

    plt.rc('text', usetex=False)
    filters_fouesnau = ['J', 'H', 'Ks']

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (24,8))

    for i, filter in enumerate(filters_fouesnau):
        y_values = cut_sample[filters_fouesnau[i]+'_sigma'].values 
        h = axs[i].hist(y_values, bins=120, histtype='step', density=False, lw=2)
        axs[i].set_xlabel(filters_fouesnau[i]+'_sigma', fontdict={'fontsize': 15}, labelpad=10)
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].set_title('{}'.format(filters_fouesnau[i]), fontdict={'fontsize': 15}, pad=20, fontweight='bold')
        axs[i].vlines(0.1, 0, np.max(h[0]), colors='red', linestyles = '--')

    fig.suptitle(suptitle, fontsize=18)

    if(is_save):
        fig_name = '/JHKs_sigma_distribution_0.1_cut'
        plt.savefig(out + fig_name +'.png', dpi=300, bbox_inches='tight', facecolor='white')


def plot_filter_uncertainties_qual(cut_sample, out, is_save = True):

    plt.rc('text', usetex=False)

    cut_sample_qual =  cut_sample[cut_sample.is_qual_true == True]

    filters_fouesnau = ['J_p50', 'H_p50', 'Ks_p50']
    filters = ['J', 'H', 'Ks']

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (24,8))

    for i, filter in enumerate(filters_fouesnau):
        y_values = cut_sample[filters[i]+'_sigma'].values 
        axs[i].hist(y_values, bins=30, histtype='step', density=False, lw=2, label='before')
        axs[i].hist(cut_sample_qual[filters[i]+'_sigma'].values , bins=30, histtype='step', density=False, lw=2, label='quality flag A or B')
        axs[i].set_xlabel(filters_fouesnau[i]+'_sigma', fontdict={'fontsize': 15}, labelpad=0)
        axs[i].legend(loc='upper right')
        axs[i].set_yscale('log')
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].set_title('{}'.format(filters_fouesnau[i]), fontdict={'fontsize': 15}, pad=20, fontweight='bold')

    fig.suptitle('Sample after the quality flag cut', fontsize=18)

    if(is_save):
        fig_name = '/JHKs_sigma_distribution_quality_cut'
        plt.savefig(out + fig_name +'.png', dpi=300, bbox_inches='tight', facecolor='white')


def plot_filter_uncertainties_w_parallax(cut_sample, out, is_save = True):

    cut_sample_parallax = cut_sample[cut_sample.parallax_sigma/cut_sample.parallax < 0.2]

    filters_fouesnau = ['J', 'H', 'Ks']

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (24,8))

    for i, filter in enumerate(filters_fouesnau):

        y_values = cut_sample[filters_fouesnau[i]+'_sigma'].values
        y_values_parallax =  cut_sample_parallax[filters_fouesnau[i]+'_sigma'].values

        axs[i].hist(y_values, bins=30, histtype='step', density=False, lw=2)
        axs[i].hist(y_values_parallax, bins=30, histtype='step', density=False, lw=2, label=r"$\sigma_\varpi/\varpi$ < 0.2")

        axs[i].set_xlabel(filters_fouesnau[i]+'_sigma', fontdict={'fontsize': 15}, labelpad=0)
        axs[i].legend(loc='upper right')
        axs[i].set_yscale('log')
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].set_title('{}'.format(filters_fouesnau[i]), fontdict={'fontsize': 15}, pad=20, fontweight='bold')

    fig.suptitle('Sample after the cut JHKS_sigma < 0.1 and sig/parallax < 0.2', fontsize=18)

    if(is_save):
        fig_name = '/JHKs_sigma_distribution_after_w_cut'
        plt.savefig(out + fig_name +'.png', dpi=300, bbox_inches='tight', facecolor='white')


    
