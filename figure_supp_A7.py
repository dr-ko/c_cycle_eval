import sys
import _shared_plot as ptool
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import numpy as np
import matplotlib.pyplot as plt
import json
from _shared import _get_set, _apply_a_mask, _apply_common_mask_g, _get_aux_data, _get_colormap_info, _get_data, _rem_nan
from string import ascii_letters as al

from _shared_regional_funcs import _get_regional_tau_c, _get_regional_tau_c_range, _get_regional_means, _get_regional_range, _get_regional_range_perFirst


def plot_variable(_ax1, _ax2, _dat_2_plot, _plotvar):

    # ------------------------------
    # plot bars
    # ------------------------------
    plt.axes(_ax1)
    dat_bar = _dat_2_plot['bar']
    dat_range = _dat_2_plot['range']
    dat_pcolor = _dat_2_plot['p_color']
    # plt.axes([0.02, 0.563, 0.951, 0.4])
    ptool.rem_axLine(['top', 'right', 'bottom'])
    ptool.rem_ticks(which_ax='x')
    plt.tick_params(labelsize=ax_fs)
    plt.bar(np.arange(len(dat_bar[:])),
            dat_bar[:],
            color=[0.8, 0.8, 0.8, 1],
            edgecolor='k',
            linewidth=0.5,
            yerr=(dat_bar[:] - dat_range[0], dat_range[1] - dat_bar[:]))

    # ------------------------------
    # write text over the bars
    # ------------------------------
    dat_obs_txt = dat_bar[:]

    for _i in range(len(dat_obs_txt)):
        plt.text(_i + 0.2,
                 dat_obs_txt[_i] + 0.05 *
                 (dat_obs_txt.max() - dat_obs_txt.min()),
                 str(round(dat_obs_txt[_i], 1)),
                 fontsize=ax_fs * 1.01,
                 weight='bold',
                 color='k',
                 ha='left',
                 va='bottom',
                 rotation=90)

    # labels and limit
    y_lab = '$Obs-based$ ' + obs_dict[_plotvar]['title'] + ' $(' + axes_list[
        _plotvar]['unit'] + ')$'  #\n' + obs_dict[cplotVar][2]
    # plt.text(0.35,0.85,y_lab, fontsize=0.9*ax_fs, transform=plt.gca().transAxes)
    h = plt.title(al[title_index] + ') ' + y_lab,
                  x=0.05,
                  y=1.04,
                  weight='bold',
                  fontsize=ax_fs * 1.3,
                  rotation=0,
                  ha='left')

    plt.ylim(axes_list[_plotvar]['range'][0], axes_list[_plotvar]['range'][1])

    # ------------------------------
    # plot the pcolor
    # ------------------------------
    plt.axes(_ax2)
    ptool.rem_axLine(['top', 'right'])

    plt.pcolor(dat_pcolor,
               cmap=cm_rat,
               edgecolors='w',
               norm=norm,
               linewidths=3)

    # ------------------------------
    # create and plot axis labels
    # ------------------------------
    xlabs = list(biomes_info.values())
    print('xlabsbe', xlabs)
    xlabs.insert(0, 'Global')
    print('xlabafe', xlabs)
    ylabs = []
    for _md in models[1:]:
        ylabs = np.append(ylabs, model_dict[_md]['model_name'])

    plt.xticks(np.arange(len(xlabs)) + 0.5,
               xlabs,
               fontsize=ax_fs,
               rotation=0,
               ma='center')
    if title_index == 0:
        plt.yticks(np.arange(len(ylabs)) + 0.5, ylabs, fontsize=ax_fs)
    else:
        plt.yticks(np.arange(len(ylabs)) + 0.5, [], fontsize=ax_fs)

    # ------------------------------
    # write text over pcolor squares
    # ------------------------------
    for _tj in range(len(ylabs)):
        for _ti in range(len(xlabs)):
            datModCk = dat_pcolor[_tj, _ti] * dat_obs_txt[_ti]
            print('the range', _plotvar, dat_range[0, _ti], dat_range[1, _ti],
                  datModCk, models[_tj], xlabs[_ti])
            if datModCk >= dat_range[0, _ti] and datModCk <= dat_range[1, _ti]:
                colotext = 'green'
            else:
                colotext = '#cc9900'
            plt.text(_ti + 0.5,
                     _tj + 0.5,
                     str(round(dat_pcolor[_tj, _ti] * dat_obs_txt[_ti], 1)),
                     fontsize=ax_fs * 1.01,
                     weight='bold',
                     color=colotext,
                     ha='center',
                     va='center',
                     bbox=dict(boxstyle="round",
                               facecolor='white',
                               edgecolor='white'))

    return


biomes_info = {
    '1': 'Arid',
    '2': 'Semi-\narid',
    '3': 'Sub-\nhumid',
    '4': 'Humid'
}

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
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas', co_settings, _co_mask=all_mask)
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _get_full_obs=True, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _get_full_obs=True, _co_mask=all_mask)
all_mod_dat_c_total = _get_data('c_total', co_settings, _get_full_obs=True, _co_mask=all_mask)

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

#
pgc_vars = co_settings['pgc_vars']

#parameters of the figure
x0 = 0.02
y0 = 0.02
wp = 0.9
#wp=1./nmodels
hp = wp
xsp = 0.02

hcolo = 0.017 * hp
wcolo = 0.3
cb_off_x = 0.1
cb_off_y = 0.15
col_m = 0

x_off = 1.05

#### get the COLORS AND COLORBARS
_bounds_rat, cm_rat, cbticks_rat, cblabels_rat = _get_colormap_info(
    'tau_c', co_settings, isratio=True)
norm = mpl.colors.BoundaryNorm(boundaries=_bounds_rat,
                               ncolors=len(_bounds_rat))

# ------------------------------
# get data for GPP and c_total
# ------------------------------
#%%%%GPP
# dat_obs_gpp = np.nanmedian(all_mod_dat_gpp['obs'], axis=0)
# dat_obs_gpp_full = all_mod_dat_gpp['obs']
# dat_obs_zonal_gpp = _get_regional_means('gpp', dat_obs_gpp, area_dat, arI,
#                                         co_settings)
# dat_obs_zonal_range_gpp = _get_regional_range_perFirst('gpp', dat_obs_gpp_full,
#                                                        area_dat, arI,
#                                                        co_settings)

dat_obs_gpp_full = all_mod_dat_gpp['obs']
dat_obs_zonal_range_gpp, dat_obs_zonal_gpp = _get_regional_range('gpp', dat_obs_gpp_full,
                                                       area_dat, arI,
                                                       co_settings)

#%%%% c_total
# dat_obs_c_total = np.nanmedian(all_mod_dat_c_total['obs'], axis=0)
# dat_obs_c_total_full = all_mod_dat_c_total['obs']
# dat_obs_zonal_c_total = _get_regional_means('c_total', dat_obs_c_total, area_dat,
#                                            arI, co_settings)
# dat_obs_zonal_range_c_total = _get_regional_range_perFirst(
#     'c_total', dat_obs_c_total_full, area_dat, arI, co_settings)

dat_obs_c_total_full = all_mod_dat_c_total['obs']
dat_obs_zonal_range_c_total, dat_obs_zonal_c_total = _get_regional_range(
    'c_total', dat_obs_c_total_full, area_dat, arI, co_settings)


dat_obs_zonal_range_tau_c, dat_obs_zonal_tau_c = _get_regional_tau_c_range(
    dat_obs_gpp_full, dat_obs_c_total_full, area_dat, arI, co_settings)

# ------------------------------
# get data for pcolor
# ------------------------------

datMod_pc_tau_c = np.zeros(((nmodels - 1, len(list(biomes_info.keys())) + 1)))
datMod_pc_gpp = np.zeros(((nmodels - 1, len(list(biomes_info.keys())) + 1)))
datMod_pc_c_total = np.zeros(((nmodels - 1, len(list(biomes_info.keys())) + 1)))
for row_m in range(1, nmodels):
    row_mod = models[row_m]
    mod_dat_row_gpp = all_mod_dat_gpp[row_mod]
    mod_dat_row_c_total = all_mod_dat_c_total[row_mod]
    m_biome_tau_c = _get_regional_tau_c(mod_dat_row_gpp, mod_dat_row_c_total,
                                      area_dat, arI, co_settings)
    m_biome_gpp = _get_regional_means('gpp', mod_dat_row_gpp, area_dat, arI,
                                      co_settings)
    m_biome_c_total = _get_regional_means('c_total', mod_dat_row_c_total,
                                         area_dat, arI, co_settings)
    datMod_pc_tau_c[row_m - 1, :] = m_biome_tau_c[:] / dat_obs_zonal_tau_c[:]
    datMod_pc_gpp[row_m - 1, :] = m_biome_gpp[:] / dat_obs_zonal_gpp[:]
    datMod_pc_c_total[row_m -
                     1, :] = m_biome_c_total[:] / dat_obs_zonal_c_total[:]

# ------------------------------
# plot the figure
# ------------------------------

fig = plt.figure(figsize=(3, 5))

axes_list = {
    'tau_c': {
        "ax1": [0.02, 0.563, 0.951, 0.4],
        "ax2": [0.05, 0.05, 0.889, 0.5],
        "range": [0, 70],
        "unit": "years"
    },
    'gpp': {
        "ax1": [x_off + 0.02, 0.563, 0.951, 0.4],
        "ax2": [x_off + 0.05, 0.05, 0.889, 0.5],
        "range": [0, 160],
        "unit": "pgC/year"
    },
    'c_total': {
        "ax1": [2 * x_off + 0.02, 0.563, 0.951, 0.4],
        "ax2": [2 * x_off + 0.05, 0.05, 0.889, 0.5],
        "range": [0, 5000],
        "unit": "pgC"
    }
}
title_index = 0
for _variable in ['tau_c', 'gpp', 'c_total']:
    dat_2_plot = {}
    dat_2_plot['bar'] = vars()['dat_obs_zonal_' + _variable]
    dat_2_plot['range'] = vars()['dat_obs_zonal_range_' + _variable]
    dat_2_plot['p_color'] = vars()['datMod_pc_' + _variable]
    ax1 = axes_list[_variable]['ax1']
    ax2 = axes_list[_variable]['ax2']
    plot_variable(ax1, ax2, dat_2_plot, _variable)
    title_index = title_index + 1

# ------------------------------
# make the colorbar
# ------------------------------
_axcol_rat = [0.9718 + 2 * x_off, 0.05, 0.0225, 0.4935]

cb = ptool.mk_colo_cont(_axcol_rat,
                        _bounds_rat,
                        cm_rat,
                        cbfs=ax_fs,
                        cb_or='vertical',
                        cbrt=0,
                        col_scale='log',
                        cbtitle='',
                        tick_locs=cbticks_rat)
cb.ax.set_yticklabels(cblabels_rat, fontsize=ax_fs, ha='left', rotation=0)
plt.ylabel('$\\mathrm{\\frac{Model}{Obs-{based}}}$',
           fontsize=ax_fs * 1.3,
           rotation=90)

# ------------------------------
# Save the figure
# ------------------------------

t_x = plt.figtext(0.5, 0.5, ' ', transform=plt.gca().transAxes)
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num +
            co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x],
            dpi=fig_set['fig_dpi'])
