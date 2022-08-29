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

from _shared import _get_set, _apply_common_mask_g, _get_aux_data, _get_data, _draw_legend_models, _fit_least_square

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.8
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
mod_colors = co_settings['model']['colors']

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

        dat_x_var = vars()['all_mod_dat_' + x_var][row_mod]
        dat_y_var = vars()['all_mod_dat_' + y_var][row_mod]

        dat_x_var, dat_y_var, mod_arI = _apply_common_mask_g(
            dat_x_var, dat_y_var, arI)
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
                     'summary_curve_fit_figure_' + fig_num + '.txt'),
        'w') as f_:
    for _ar in pcoffs_ar:
        print(_ar[:])
        f_.write(_ar[:] + '\n')

#-------------------------------------------------------------------
# plot the figure
#-------------------------------------------------------------------

fig = plt.figure(figsize=(6.8, 6.8))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
reInd = 0
for rel in rela_tions.keys():
    x_var = rel.split('-')[0]
    y_var = rel.split('-')[1]
    y_tit = obs_dict[y_var]['title'] + ' (' + obs_dict[y_var][
            'unit'] + ')'
    x_tit = obs_dict[x_var]['title'] + ' (' + obs_dict[x_var][
            'unit'] + ')'

    for ariName in aridity_list:
        arInd = aridity_list.index(ariName)
        spInd = 4 * reInd + arInd + 1
        print(rel, ariName, spInd)
        plt.subplot(4, 4, spInd)
        ptool.ax_orig(axfs=ax_fs * 0.9)
        h = plt.text(0.02,
                     0.97,
                     al[spInd - 1],
                     weight='bold',
                     fontsize=ax_fs,
                     rotation=0,
                     transform=plt.gca().transAxes)
        for row_m in range(nmodels):
            row_mod = models[row_m]
            x_dat = fit_dict[rel][ariName][row_mod]['xx']
            y_dat = fit_dict[rel][ariName][row_mod]['yy']
            plt.xlim(obs_dict[x_var]['plot_range_fit'][0],
             obs_dict[x_var]['plot_range_fit'][1])
            plt.ylim(obs_dict[y_var]['plot_range_fit'][0],
             obs_dict[y_var]['plot_range_fit'][1])
            if row_mod == 'obs':
                _lw = 1.25
                mName = 'Obs-based'
            else:
                _lw = 0.65
                mName = model_dict[row_mod]['model_name']
            if row_mod == 'obs':
                plt.plot(x_dat,
                            y_dat,
                            ls='--',
                            lw=_lw,
                            color=mod_colors[row_mod],
                            marker=None,
                            label=mName, zorder=10)
            else:
                plt.plot(x_dat,
                            y_dat,
                            ls='-',
                            lw=_lw,
                            color=mod_colors[row_mod],
                            marker=None,
                            label=mName)
        if rela_tions[rel]['log_y']:
            plt.gca().set_yscale('log')
            ptool.put_ticks(nticks=4, which_ax='x')
        else:
            ptool.put_ticks(nticks=4, which_ax='both')
        # plt_subplot(x_dat,y_dat,_logY=_logY)
        plt.plot(x_dat,
                    y_dat * np.nan,
                    ls='-',
                    lw=_lw,
                    color=mod_colors[row_mod],
                    marker=None,
                    label='',
                    zorder=0)
        if arInd == 0 and reInd == 0:
            leg = _draw_legend_models(co_settings,
                                        loc_a=(0.6722, 1.2217125855), ax_fs = 0.95*ax_fs, inc_mme = False)
        if reInd == 0:
            plt.title(aridity_list[arInd], y=0.98, fontsize=ax_fs * 0.9)
        if arInd == 0:
            #     if reInd == 1 or reInd ==3:
            plt.ylabel(y_tit, fontsize=ax_fs * 0.9, color='#666666')
        plt.xlabel(x_tit, fontsize=ax_fs * 0.9, color='#666666')
    reInd = reInd + 1

plt.savefig(co_settings['fig_settings']['fig_dir'] + 'xtra_fig_' + fig_num +
            co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])
