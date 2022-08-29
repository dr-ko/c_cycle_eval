import sys
import _shared_plot as ptool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
from string import ascii_letters as al
from _shared import _get_set, _apply_common_mask_g, _get_aux_data, _get_data, _draw_legend_models, _plot_correlations


#-------------------------------------------
# ZONAL MEAN OF THE TAU
#-------------------------------------------
def get_zonal_tau_c(_datgpp, _datc_total, _area_dat, zonal_set):
    _lats = zonal_set['lats']
    bandsize_mean = zonal_set['bandsize_mean']
    min_points = zonal_set['min_points']

    _latint = abs(_lats[1] - _lats[0])
    __dat = np.ones((np.shape(_datgpp)[0])) * np.nan
    windowSize = int(np.round(bandsize_mean / (_latint * 2)))

    # remove ocean-only latitude
    _datgpp, _datc_total = _apply_common_mask_g(_datgpp, _datc_total)
    v_mask = ~np.ma.masked_invalid(_datgpp).mask * ~np.ma.masked_invalid(
        _datc_total).mask
    nvalids = np.sum(v_mask, axis=1)
    lat_indices = np.argwhere(nvalids > 0)

    # loop over latitude
    for _latInd in lat_indices:
        li = _latInd[0]
        istart = max(0, li - windowSize)
        iend = min(np.size(_lats), li + windowSize + 1)
        _areaZone = _area_dat[istart:iend, :]
        _gppZone = _datgpp[istart:iend, :] * _areaZone
        _c_totalZone = _datc_total[istart:iend, :] * _areaZone
        v_mask = ~np.ma.masked_invalid(_gppZone).mask * ~np.ma.masked_invalid(
            _c_totalZone).mask
        nvalids = np.sum(v_mask)
        if nvalids > min_points:
            __dat[li] = np.nansum(_c_totalZone) / np.nansum(_gppZone)
    return __dat


def get_zonal_tau_c_percentiles(_obsgppFull, _obsc_totalFull, _area_dat,
                              zonal_set):
    perc_range = zonal_set['perc_range']
    nMemb_gpp = len(_obsgppFull)
    nMemb_c_total = len(_obsc_totalFull)
    nMemb = nMemb_gpp * nMemb_c_total
    # nMemb = len(_obsgppFull)
    # nMemb = 2
    nLats = np.shape(_obsgppFull[0])[0]
    _perFull = np.zeros((nLats, nMemb))
    memb_index = 0
    for memb_gpp in range(nMemb_gpp):
        gpp_memb = _obsgppFull[memb_gpp]
        for memb_c_total in range(nMemb_c_total):
            c_total_memb = _obsc_totalFull[memb_c_total]
            zoneMemb = get_zonal_tau_c(gpp_memb, c_total_memb, area_dat, zonal_set)
            _perFull[:, memb_index] = zoneMemb[:]
            memb_index = memb_index + 1
    # for memb in range(nMemb):
    #     print('zonal tau:', memb)
    #     corr_zone = get_zonal_tau_c(_obsgppFull[memb], _obsc_totalFull[memb],
    #                               _area_dat, zonal_set)
    #     _perFull[:, memb] = corr_zone
    dat_5 = np.nanpercentile(_perFull, perc_range[0], axis=1)
    dat_median = np.nanpercentile(_perFull, 50, axis=1)
    dat_95 = np.nanpercentile(_perFull, perc_range[1], axis=1)
    return dat_5, dat_95, dat_median


#-------------------------------------------
## Zonal means of gpp and ctoal
#-------------------------------------------
def get_zonal_gpp(_datgpp, _area_dat, zonal_set):
    _lats = zonal_set['lats']
    bandsize_mean = zonal_set['bandsize_mean']
    min_points = zonal_set['min_points']
    _latint = abs(_lats[1] - _lats[0])
    __dat = np.ones((np.shape(_datgpp)[0])) * np.nan
    windowSize = int(np.round(bandsize_mean / (_latint * 2)))
    v_mask = ~np.ma.masked_invalid(_datgpp).mask
    nvalids = np.sum(v_mask, axis=1)
    lat_indices = np.argwhere(nvalids > 0)
    for _latInd in lat_indices:
        li = _latInd[0]
        istart = max(0, li - windowSize)
        iend = min(np.size(_lats), li + windowSize + 1)
        _areaZone = _area_dat[istart:iend, :]
        _gppZone = _datgpp[istart:iend, :] * _areaZone
        v_mask = ~np.ma.masked_invalid(_gppZone).mask
        nvalids = np.sum(v_mask)
        if nvalids > min_points:
            __dat[li] = np.nansum(_gppZone) / np.nansum(_areaZone)
    return __dat


def get_zonal_gpp_percentiles_pf(_obsData, _area_dat, zonal_set):
    perc_range = zonal_set['perc_range']
    dat_5 = np.nanpercentile(_obsData, perc_range[0], axis=0)
    dat_median = np.nanpercentile(_obsData, 50, axis=0)
    dat_95 = np.nanpercentile(_obsData, perc_range[1], axis=0)
    dat_zonal_5 = get_zonal_gpp(dat_5, _area_dat, zonal_set)
    dat_zonal_median = get_zonal_gpp(dat_median, _area_dat, zonal_set)
    dat_zonal_95 = get_zonal_gpp(dat_95, _area_dat, zonal_set)
    return dat_zonal_5, dat_zonal_95, dat_zonal_median



def get_zonal_gpp_percentiles(_obsData, _area_dat, zonal_set):
    perc_range = zonal_set['perc_range']
    nMemb_gpp = len(_obsData)
    nMemb = nMemb_gpp
    # nMemb = len(_obsgppFull)
    # nMemb = 2
    nLats = np.shape(_obsData[0])[0]
    _perFull = np.zeros((nLats, nMemb))
    memb_index = 0
    for memb_gpp in range(nMemb):
        gpp_memb = _obsData[memb_gpp]
        zoneMemb = get_zonal_gpp(gpp_memb, _area_dat, zonal_set)
        _perFull[:, memb_gpp] = zoneMemb[:]
    dat_5 = np.nanpercentile(_perFull, perc_range[0], axis=1)
    dat_median = np.nanpercentile(_perFull, 50, axis=1)
    dat_95 = np.nanpercentile(_perFull, perc_range[1], axis=1)
    return dat_5, dat_95, dat_median

#-------------------------------------------
# plotting
#-------------------------------------------


def plot_zonal_var(_sp, plot_var, all_gpp, all_c_total, area_dat, zonal_set,
                   fig_set):
    if plot_var == 'tau_c':
        dat_obs_zonal_5, dat_obs_zonal_95, dat_obs_zonal = get_zonal_tau_c_percentiles(
            all_gpp['obs'], all_c_total['obs'], area_dat, zonal_set)
    elif plot_var == 'gpp':
        dat_obs_zonal_5, dat_obs_zonal_95, dat_obs_zonal = get_zonal_gpp_percentiles(
            all_gpp['obs'], area_dat, zonal_set)
    else:
        dat_obs_zonal_5, dat_obs_zonal_95, dat_obs_zonal = get_zonal_gpp_percentiles(
            all_c_total['obs'], area_dat, zonal_set)

    _sp.plot(dat_obs_zonal,
             zonal_set['lats'],
             color='k',
             lw=fig_set['lwMainLine'],
             label='Obs-based',
             zorder=10)
    _sp.fill_betweenx(zonal_set['lats'],
                      dat_obs_zonal_5,
                      dat_obs_zonal_95,
                      facecolor='grey',
                      alpha=0.40)

    # plot the tau from each model
    plt.gca().tick_params(labelsize=ax_fs * 0.91)

    zonal_mm_ens = np.ones((nmodels - 1, np.shape(all_mask)[0])) * np.nan
    modI = 0
    for row_m in range(1, nmodels):
        row_mod = models[row_m]
        if plot_var == 'tau_c':
            dat_mod = get_zonal_tau_c(all_gpp[row_mod], all_c_total[row_mod],
                                    area_dat, zonal_set)
        elif plot_var == 'gpp':
            dat_mod = get_zonal_gpp(all_gpp[row_mod], area_dat, zonal_set)
        else:
            dat_mod = get_zonal_gpp(all_c_total[row_mod], area_dat, zonal_set)

        zonal_mm_ens[modI] = dat_mod
        modI = modI + 1
        _sp.plot(np.ma.masked_equal(dat_mod, np.nan),
                 zonal_set['lats'],
                 color=mod_colors[row_mod],
                 lw=fig_set['lwModLine'],
                 label=model_dict[row_mod]['model_name'])

    # get the multimodel ensemble tau from multimodel ensemble gpp and c_total
    zonal_mm_ens = np.nanmedian(zonal_mm_ens, axis=0)
    _sp.plot(np.ma.masked_equal(zonal_mm_ens, np.nan),
             zonal_set['lats'],
             color='blue',
             lw=fig_set['lwMainLine'],
             label='Model Ensemble',
                 zorder=9)


    # set the scale and ranges of axes
    valrange_md = obs_dict[plot_var]['plot_range_zonal']
    if plot_var in ['tau_c', 'c_total']:
        plt.gca().set_xscale('log')
    plt.xlim(valrange_md[0], valrange_md[1])
    plt.ylim(-60, 85)
    plt.axhline(y=0, lw=0.48, color='grey')
    return


#parameters of the figure

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']
nmodels = len(models)

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.8
co_settings['fig_settings']['ax_fs'] = ax_fs
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

fig_set['ax_fs'] = ax_fs
fig_set['lwMainLine'] = 1.03
fig_set['lwModLine'] = 0.45
fig_set['mod_colors'] = co_settings['model']['colors']

zonal_set = fig_set['zonal']
zonal_set['lats'] = np.linspace(-89.75, 89.75, 360, endpoint=True)[::-1]

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_tasp = _get_data('tasp', co_settings, _co_mask=all_mask)
all_tau_c = _get_data('tau_c', co_settings, _get_full_obs=True)
all_gpp = _get_data('gpp', co_settings, _get_full_obs=True)
all_c_total = _get_data('c_total', co_settings, _get_full_obs=True)
all_pr['obs'] = all_pr['lpj']
all_tasp['obs'] = all_tasp['lpj']

all_data = {}
all_data['pr'] = all_pr
all_data['tasp'] = all_tasp
all_data['tau_c'] = all_tau_c
all_data['c_total'] = all_c_total
all_data['gpp'] = all_gpp

mod_colors = co_settings['model']['colors']

# define figure
fig = plt.figure(figsize=(5.5, 9))
plt.subplots_adjust(hspace=0.2, wspace=0.3)
plt.gca().tick_params(labelsize=ax_fs * 0.91)
tit_x = 0.25
tit_y = 1.0
#-------------------------------------------
# Loop through the variables
#-------------------------------------------
variables = 'tau_c gpp c_total'.split()
# plot observation tau
spInd = 1
for _variable in variables:
    # zonal means
    sp1 = plt.subplot(3, 3, spInd)
    plot_zonal_var(sp1, _variable, all_gpp, all_c_total, area_dat, zonal_set,
                   fig_set)
    if spInd == 1:
        leg = _draw_legend_models(co_settings, loc_a=(-0.1144, 1.13))

    h = plt.title(al[spInd - 1] + ') ' + obs_dict[_variable]['title'] + ' (' +
                  obs_dict[_variable]['unit'] + ')',
                  x=tit_x,
                  y=tit_y,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    plt.ylabel('Latitude ($^\\circ N$)', fontsize=ax_fs, ma='center')
    ptool.rem_axLine(['top', 'right'])

    # zonal MAT correlation controlled for MAP
    sp2 = plt.subplot(3, 3, spInd + 1)
    var_info = {}
    var_info['x'] = _variable
    var_info['y'] = 'tasp'
    var_info['z'] = ['pr']
    var_name = obs_dict[_variable]['title']
    if "$" in var_name:
        var_name = var_name.replace("$", "")
    x_lab = '$r_{' + var_name + '-MAT,MAP}$'
    h = plt.title(al[spInd] + ') ' + x_lab,
                  x=tit_x,
                  y=tit_y,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    ptool.rem_ticklabels(which_ax='y')
    plt.gca().tick_params(labelsize=ax_fs * 0.91)
    _plot_correlations(sp2, all_data, var_info, zonal_set, fig_set,
                       co_settings)

    # zonal MAP correlation controlled for MAT
    sp3 = plt.subplot(3, 3, spInd + 2)
    var_info = {}
    var_info['x'] = _variable
    var_info['y'] = 'pr'
    var_info['z'] = ['tasp']
    x_lab = '$r_{' + var_name + '-MAP,MAT}$'
    h = plt.title(al[spInd + 1] + ') ' + x_lab,
                  x=tit_x,
                  y=tit_y,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    ptool.rem_ticklabels(which_ax='y')
    plt.gca().tick_params(labelsize=ax_fs * 0.91)
    if _variable == 'tau_c':
        corr_dat = all_tau_c
    elif _variable == 'gpp':
        corr_dat = all_gpp
    else:
        corr_dat = all_c_total
    _plot_correlations(sp3, all_data, var_info, zonal_set, fig_set,
                       co_settings)
    spInd = spInd + len(variables)
#-------------------------------------------
# save the figure
#-------------------------------------------
t_x = plt.figtext(0.96, 1.038, ' ', transform=plt.gca().transAxes)
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x, leg],
            dpi=fig_set['fig_dpi'])
plt.close(1)
