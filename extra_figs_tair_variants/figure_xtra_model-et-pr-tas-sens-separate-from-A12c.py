import sys, os
from typing import OrderedDict
import _shared_plot as ptool
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import json
from string import ascii_letters as al

from _shared import _get_set, _apply_common_mask_g, _get_aux_data, _get_data, _draw_legend_aridity, _fit_least_square

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.7
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_taspn = _get_data('taspn', co_settings, _co_mask=all_mask)
all_mod_dat_evapotrans = _get_data('evapotrans', co_settings, _co_mask=all_mask)
all_mod_dat_taspn['obs'] = all_mod_dat_taspn['lpj']
all_mod_dat_pr['obs'] = all_mod_dat_pr['lpj']

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

fit_dict = {}

rela_tions = OrderedDict({
    'taspn-evapotrans': {
        'p_bounds': [(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': False,
        'inter_cept': True,
        'label_y': True,
        'label_x': True,
        'inset_x': 0.13951,
        'inset_y': 0.625627625
    },
    'pr-evapotrans': {
        'p_bounds': [(-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': False,
        'inter_cept': False,
        'label_y': False,
        'label_x': True,
        'inset_x': 0.13951,
        'inset_y': 0.625627625
    }
})

fit_method = co_settings['fig_settings']['fit_method']

#-------------------------------------------------------------------
# get the relationships for all models
#-------------------------------------------------------------------
pcoffs_ar = []
nmodels = len(models)

for rel in rela_tions.keys():
    x_var = rel.split('-')[0]
    y_var = rel.split('-')[1]
    fit_dict[rel] = {}
    for row_m in range(nmodels):
        row_mod = models[row_m]
        print(rel, ':', row_mod)

        dat_x_var = vars()['all_mod_dat_' + x_var]['lpj'].copy()
        dat_y_var = vars()['all_mod_dat_' + y_var][row_mod].copy()

        dat_x_var, dat_y_var, mod_arI = _apply_common_mask_g(
            dat_x_var, dat_y_var, arI.copy())
        if row_mod == 'obs':
            pcoffs = rel + '|obs|'
        else:
            pcoffs = rel + '|' + model_dict[row_mod]['model_name'] + '|'

        # loop through aridity classes
        for _tr in range(len(_aridity_bounds) - 1):
            taspn1 = _aridity_bounds[_tr]
            taspn2 = _aridity_bounds[_tr + 1]
            mod_x_tmp = np.ma.masked_inside(mod_arI, taspn1, taspn2).mask
            dat_y_tr = dat_y_var[mod_x_tmp]
            dat_x_tr = dat_x_var[mod_x_tmp]
            ariName = aridity_list[_tr]
            if ariName not in list(fit_dict[rel].keys()):
                fit_dict[rel][ariName] = {}
            if row_mod not in list(fit_dict[rel][ariName].keys()):
                fit_dict[rel][ariName][row_mod] = {}
            # fit for a given model and aridity
            fit_dat = _fit_least_square(
                dat_x_tr,
                dat_y_tr,
                _logY=rela_tions[rel]['log_y'],
                _intercept=rela_tions[rel]['inter_cept'],
                method=fit_method,
                _bounds=rela_tions[rel]['p_bounds'])

            # create the string to write in summary text file
            coffs = fit_dat['coef']
            r2 = fit_dat['metr']['r2']
            r_mad = fit_dat['metr']['r_mad']

            pcoff = "%.2e" % coffs[0] + '|' + "%.2e" % coffs[1] + '|' + str(
                np.round(coffs[2],
                         2)) + '|' '%.2f' % r2 + '|' '%.2f' % r_mad + ''

            xx = fit_dat['pred']['x']
            yy = fit_dat['pred']['y']

            # save the fitted data in dictionary to use for plotting later
            fit_dict[rel][ariName][row_mod]['coffs'] = coffs
            fit_dict[rel][ariName][row_mod]['xx'] = xx
            fit_dict[rel][ariName][row_mod]['yy'] = yy
            pcoffs = pcoffs + pcoff + '|'
        pcoffs_ar = np.append(pcoffs_ar, pcoffs)

# process the strings and arrays to write to the summary file
pcoffs_ar = np.array(pcoffs_ar).reshape(-1, 4)
pcoffs_ar = pcoffs_ar.flatten(order='F')

with open(
        os.path.join(co_settings['fig_settings']['fig_dir'],
                     'summary_curve_fit_figure_' + fig_num +
            co_settings['exp_suffix'] + '.txt'),
        'w') as f_:
    for _ar in pcoffs_ar:
        print(_ar[:])
        f_.write(_ar[:] + '\n')

#-------------------------------------------------------------------
# plot the figure
#-------------------------------------------------------------------

fig = plt.figure(figsize=(3, 7.8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# loop through models
for row_m in range(1, nmodels):
    # loop through relationships
    reInd = 0
    for rel in rela_tions.keys():
        spInd = 2 * (row_m - 1) + reInd + 1
        plt.subplot(nmodels - 1, 2, spInd)
        h = plt.text(0.02,
                     0.97,
                     al[spInd - 1],
                     weight='bold',
                     fontsize=ax_fs,
                     rotation=0,
                     transform=plt.gca().transAxes)
        x_var = rel.split('-')[0]
        y_var = rel.split('-')[1]
        y_tit = obs_dict[y_var]['title'] + ' (' + obs_dict[y_var][
                'unit'] + ')'
        x_tit = obs_dict[x_var]['title'] + ' (' + obs_dict[x_var][
                'unit'] + ')'

        for ariName in aridity_list:
            # arInd = aridity_list.index(ariName)
            row_mod = models[row_m]
            x_dat = fit_dict[rel][ariName][row_mod]['xx']
            y_dat = fit_dict[rel][ariName][row_mod]['yy']
            coffs = fit_dict[rel][ariName][row_mod]['coffs']
            x_dat_obs = fit_dict[rel][ariName]['obs']['xx']
            y_dat_obs = fit_dict[rel][ariName]['obs']['yy']
            coffs_obs = fit_dict[rel][ariName]['obs']['coffs']
            arInd = aridity_list.index(ariName)
            print(row_mod, rel, ariName, spInd)
            ptool.ax_orig(axfs=ax_fs * 0.8)

            plt.xlim(obs_dict[x_var]['plot_range_fit'][0],
             obs_dict[x_var]['plot_range_fit'][1])
            plt.ylim(obs_dict[y_var]['plot_range_fit'][0],
             obs_dict[y_var]['plot_range_fit'][1])

            if row_mod == 'obs':
                _lw = 0.95
                mName = 'Obs-based'
            else:
                _lw = 0.95
                mName = model_dict[row_mod]['model_name']
            plt.plot(x_dat_obs,
                     y_dat_obs,
                     ls='--',
                     lw=0.5 * _lw,
                     color=color_list[arInd],
                     marker=None,
                     label=ariName)
            plt.plot(x_dat,
                     y_dat,
                     ls='-',
                     lw=0.8 * _lw,
                     color=color_list[arInd],
                     marker=None,
                     label=None,
                     zorder=0)

            met = str(np.abs(np.round(coffs[1] / coffs_obs[1], 2)))
            met2 = ''

            if rel == 'pr-evapotrans':
                met2 = ''
            else:
                if rela_tions[rel]['log_y']:
                    met2 = ', ' + str(
                        np.abs(np.round(10**coffs[2] / 10**coffs_obs[2], 2)))
                else:
                    met2 = ', ' + str(np.abs(np.round(coffs[2] / coffs_obs[2], 2)))
            plt.text(rela_tions[rel]['inset_x'],
                     rela_tions[rel]['inset_y'] + arInd * 0.118,
                     met + met2,
                     color=color_list[arInd],
                     fontsize=0.772585 * ax_fs,
                     transform=plt.gca().transAxes)

        if rela_tions[rel]['log_y']:
            plt.gca().set_yscale('log')
            ptool.put_ticks(nticks=4, which_ax='x')
        else:
            ptool.put_ticks(nticks=4, which_ax='both')

        if row_m == 1 and reInd == 0:
            leg = _draw_legend_aridity(co_settings,
                                        loc_a=(0.38227722, 1.37125855), ax_fs=ax_fs*0.87)
        if row_m == 1:
            plt.title(y_tit, y=1.05, fontsize=ax_fs * 0.9)
        if reInd == 1:
            h = plt.ylabel(
                mName,
                color=co_settings['model']['colors'][row_mod],
                # weight='bold',
                fontsize=ax_fs * 0.928,
                rotation=90)
            plt.gca().yaxis.set_label_position("right")
        if row_m == nmodels - 1:
            plt.xlabel(x_tit, fontsize=ax_fs * 0.9)
        reInd = reInd + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num +
            co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])

