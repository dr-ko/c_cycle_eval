import sys
from typing import OrderedDict
import _shared_plot as ptool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
from string import ascii_letters as al
from _shared import _get_set, _apply_common_mask_g, _get_aux_data, _get_data, _draw_legend_aridity, _fit_least_square


def fit_and_plot(_xDat,
                 _yDat,
                 _logY=False,
                 intercept=True,
                 _fit_method='quad',
                 bounds=None):

    ptool.ax_orig(axfs=ax_fs * 0.8)
    im = plt.scatter(_xDat,
                     _yDat,
                     s=0.03,
                     c=mod_arI,
                     cmap=cm_rat,
                     norm=mpl.colors.BoundaryNorm(_aridity_bounds,
                                                  len(_aridity_bounds)),
                     linewidths=0.3218,
                     alpha=0.274)
    for _tr in range(len(_aridity_bounds) - 1):
        tas1 = _aridity_bounds[_tr]
        tas2 = _aridity_bounds[_tr + 1]
        mod_tas_tmp = np.ma.masked_inside(mod_arI, tas1, tas2).mask
        _yDat_tr = _yDat[mod_tas_tmp]
        _xDat_tr = _xDat[mod_tas_tmp]

        print('Fitting for:')
        print('             ', aridity_list[_tr])
        fit_dat = _fit_least_square(_xDat_tr,
                                    _yDat_tr,
                                    _logY=_logY,
                                    method=_fit_method,
                                    _intercept=intercept,
                                    _bounds=bounds)

        plt.plot(fit_dat['pred']['x'],
                 fit_dat['pred']['y'],
                 c=color_list[_tr],
                 ls='-',
                 lw=0.95,
                 marker=None,
                 label=aridity_list[_tr])
        if _logY:
            plt.gca().set_yscale('log')
        pcoff = fit_dat['coef']
        r2 = fit_dat['metr']['r2']
        r_mad = fit_dat['metr']['r_mad']

        if inset_info == 'all':
            strW = "%.2e" % pcoff[0] + '|' + "%.2e" % pcoff[1] + '|' + str(
                np.round(pcoff[2], 2)
            ) + ' ($r^2$=' '%.2f' % r2 + '|' + '$r_{mad}$=' '%.2f' % r_mad + ')'
            inset_x = 0.1151
            f_scale = 0.585
        elif inset_info == 'coef':
            strW = "%.2e" % pcoff[0] + '|' + "%.2e" % pcoff[1] + '|' + str(
                np.round(pcoff[2], 2))
            inset_x = 0.5051
            f_scale = 0.71
        elif inset_info == 'metr':
            strW = '$r^2$=' '%.2f' % r2 + '|' + '$r_{mad}$=' '%.2f' % r_mad
            inset_x = 0.55151
            f_scale = 0.71
        else:
            strW = ''
            inset_x = 0.5
            f_scale = 0.71
        plt.text(inset_x,
                 rela_tions[rel]['inset_y'] + _tr * 0.05,
                 strW,
                 color=color_list[_tr],
                 fontsize=f_scale * ax_fs,
                 transform=plt.gca().transAxes)

    return im


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs']
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs
mask_all = 'model_valid_tau_cObs'.split()

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tasp = _get_data('tasp', co_settings, _co_mask=all_mask)
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask)
all_mod_dat_c_total = _get_data('c_total', co_settings, _co_mask=all_mask)
all_mod_dat_tasp['obs'] = all_mod_dat_tasp['lpj']
all_mod_dat_pr['obs'] = all_mod_dat_pr['lpj']
models = 'obs'.split()
nmodels = len(models)

fit_method = co_settings['fig_settings']['fit_method']
inset_info = 'coef'  # can be metr for metrics only, or all for everything

rela_tions = OrderedDict({
    'tasp-tau_c': {
        'p_bounds': [(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': True,
        'inter_cept': True,
        'label_y': True,
        'label_x': False,
        'inset_x': 0.627,
        'inset_y': 0.827625
    },
    'pr-tau_c': {
        'p_bounds': [(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': True,
        'inter_cept': True,
        'label_y': False,
        'label_x': False,
        'inset_x': 0.627,
        'inset_y': 0.827625
    },
    'tasp-gpp': {
        'p_bounds': [(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': False,
        'inter_cept': True,
        'label_y': True,
        'label_x': True,
        'inset_x': 0.13951,
        'inset_y': 0.827625
    },
    'pr-gpp': {
        'p_bounds': [(-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)],
        'log_y': False,
        'inter_cept': False,
        'label_y': False,
        'label_x': True,
        'inset_x': 0.13951,
        'inset_y': 0.827625
    }
})

fig = plt.figure(figsize=(6, 6))

plt.subplots_adjust(hspace=0.2, wspace=0.2)
sp_index = 1
for rel in rela_tions.keys():
    x_var = rel.split('-')[0]
    y_var = rel.split('-')[1]
    plt.subplot(2, 2, sp_index)
    for row_m in range(nmodels):
        print('------------------------------------------------------')
        row_mod = models[row_m]
        print(rel, ':', row_mod)

        dat_x_var = vars()['all_mod_dat_' + x_var][row_mod]
        dat_y_var = vars()['all_mod_dat_' + y_var][row_mod]

        dat_x_var, dat_y_var, mod_arI = _apply_common_mask_g(
            dat_x_var, dat_y_var, arI)
        fit_and_plot(dat_x_var,
                     dat_y_var,
                     _logY=rela_tions[rel]['log_y'],
                     intercept=rela_tions[rel]['inter_cept'],
                     _fit_method=fit_method,
                     bounds=rela_tions[rel]['p_bounds'])
        h = plt.title(al[sp_index - 1],
                      weight='bold',
                      x=0.061,
                      y=0.95,
                      fontsize=ax_fs,
                      rotation=0)
    if sp_index == 1:
        leg = _draw_legend_aridity(co_settings, loc_a=(1.2422, 1.0625855))
    if rela_tions[rel]['label_y']:
        plt.ylabel(obs_dict[y_var]['title'] + ' (' + obs_dict[y_var]['unit'] +
                   ')',
                   ha='center',
                   fontsize=ax_fs * 0.9)
    if rela_tions[rel]['label_x']:
        plt.xlabel(obs_dict[x_var]['title'] + ' (' + obs_dict[x_var]['unit'] +
                   ')',
                   ha='center',
                   fontsize=ax_fs * 0.9)
    plt.xlim(obs_dict[x_var]['plot_range_fit'][0],
             obs_dict[x_var]['plot_range_fit'][1])
    plt.ylim(obs_dict[y_var]['plot_range_fit'][0],
             obs_dict[y_var]['plot_range_fit'][1])
    if rela_tions[rel]['log_y']:
        ptool.put_ticks(nticks=4, which_ax='x')
    else:
        ptool.put_ticks(nticks=4, which_ax='both')
    # ptool.put_ticks(nticks=4, which_ax='both')
    sp_index = sp_index + 1


plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num +
            co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])
plt.close()
