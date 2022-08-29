import _shared_plot as ptool
from typing import OrderedDict
import os
import sys
import numpy as np
from string import ascii_letters as al
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import matplotlib.pyplot as plt
import h5py
from _shared import _get_set, _apply_a_mask, _apply_common_mask_g,_get_aux_data, _get_data, _draw_legend_aridity, _fit_least_square
import xarray as xr
import cartopy.crs as ccrs
#--------


def get_fitted_data(_xDat,
                    _yDat,
                    _logY=False,
                    intercept=True,
                    _fit_method='quad',
                    bounds=[(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)]):

    pred_o = {}
    c_alpha = 0.9915
    for _tr in range(len(_aridity_bounds) - 1):
        pred_o[_tr + 1] = {}
        tas1 = _aridity_bounds[_tr]
        tas2 = _aridity_bounds[_tr + 1]
        mod_tas_tmp = np.ma.masked_inside(mod_arI, tas1, tas2).mask
        _yDat_tr = _yDat[mod_tas_tmp]
        _xDat_tr = _xDat[mod_tas_tmp]
        fit_dat = _fit_least_square(_xDat_tr,
                                    _yDat_tr,
                                    _logY=_logY,
                                    method=_fit_method,
                                    _intercept=intercept,
                                    _bounds=bounds)

        pred_o[_tr + 1]['x'] = fit_dat['pred']['x']
        pred_o[_tr + 1]['y'] = fit_dat['pred']['y']
        plt.plot(fit_dat['pred']['x'],
                fit_dat['pred']['y'],
                c=color_list[_tr],
                ls='-',
                lw=0.48225,
                marker=None,
                # label=aridity_list[_tr],
                alpha=c_alpha)

    return pred_o


def plt_median(full_pred, spStart, nPred):

    for rel_n in range(1, 5):
        firstTime=True
        fig = plt.figure(figsize=(7, 7))
        plt.subplots_adjust(hspace=0.32, wspace=0.32)
        plt.subplot(4, 4, spStart + rel_n - 1)
        for ar in range(1, 5):
            full_dat_ar_x = np.nanmedian(
                full_pred[rel_n][ar]['x'].reshape(nPred, -1), 0)

            full_dat_ar_y = np.nanmedian(
                full_pred[rel_n][ar]['y'].reshape(nPred, -1), 0)
            full_dat_ar_y_25 = np.nanpercentile(
                full_pred[rel_n][ar]['y'].reshape(nPred, -1), 25, axis=0)
            full_dat_ar_y_75 = np.nanpercentile(
                full_pred[rel_n][ar]['y'].reshape(nPred, -1), 75, axis=0)
            plt.plot(full_dat_ar_x,
                     full_dat_ar_y,
                     c=color_list[ar - 1],
                     ls='-',
                     lw=1.15,
                     marker=None,
                     alpha=1)
            plt.fill_between(full_dat_ar_x,
                             full_dat_ar_y_25,
                             full_dat_ar_y_75,
                             facecolor=color_list[ar - 1],
                             alpha=0.25)

        for _ar in range(len(aridity_list)):
            plt.plot(full_dat_ar_x * np.nan,
                    full_dat_ar_y * np.nan,
                    c=color_list[_ar],
                    ls='-',
                    lw=1.15,
                    marker=None,
                    label=aridity_list[_ar],
                    alpha=1)
        if firstTime:
            leg = _draw_legend_aridity(co_settings, loc_a=(-1.02422, 1.21625855))
            firstTime = False


        plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + str(rel_n) +'.'+
                    fig_set['fig_format'],
                    bbox_inches='tight',
                    bbox_extra_artists=[leg],
                    dpi=fig_set['fig_dpi'])
    return


def get_full_d():
    full_pred = {}
    for rel_n in range(1, 5):
        full_pred[rel_n] = {}
        for ar in range(1, 5):
            full_pred[rel_n][ar] = {}
            full_pred[rel_n][ar]['x'] = []
            full_pred[rel_n][ar]['y'] = []
    return full_pred


def fill_pred_d(full_pred, part_pred, key):
    rel_n = key
    for ar in range(1, 5):
        full_pred[rel_n][ar]['x'] = np.append(full_pred[rel_n][ar]['x'],
                                            part_pred[ar]['x'])
        full_pred[rel_n][ar]['y'] = np.append(full_pred[rel_n][ar]['y'],
                                            part_pred[ar]['y'])
    return full_pred

def make_map(_ax, _whichVar, _dat):
    plt.imshow(np.ma.masked_less(_dat.reshape(-1, 360, 720).mean(0)[:300, :], -999.),
            vmin=co_settings['obs_dict'][_whichVar]['plot_range_map'][0],
            vmax=co_settings['obs_dict'][_whichVar]['plot_range_map'][1],
            cmap=co_settings['obs_dict'][_whichVar]['color_scale'],
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -60, 90])
    _ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    _ax.coastlines(linewidth=0.4, color='grey')
    plt.gca().outline_patch.set_visible(False)
    plt.colorbar(pad=0.0316 ,orientation='vertical', shrink = 0.676, extend='both')
    plt.suptitle(_whichVar +' ('+co_settings['obs_dict'][_whichVar]['title']+', '+co_settings['obs_dict'][_whichVar]['unit']+')', y=0.9094,x=0.576, ha='center')
    return

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.8076
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs
mask_all = 'model_valid_tau_cObs'.split()

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

all_mask, arI, area_dat = _get_aux_data(co_settings)

models = 'obs'.split()
nmodels = len(models)
# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas',
                                          co_settings,
                                          _co_mask=all_mask)
#---------------------------------------------------------------------------
# get the observations (Carvalhais, 2014) and add them to the full cube
#---------------------------------------------------------------------------
co_settings['model']['names'] = models

# create a copy of the precipitation used as observation
mod_tas_0 = all_mod_dat_tas['lpj']
mod_pr_0 = all_mod_dat_pr['lpj']


#---------------------------------------------------------------------------
# get the additional observations (FLUXCOM GPP)
#---------------------------------------------------------------------------
mod_dat_f = xr.open_dataset(os.path.join(top_dir, obs_dict['gpp']['obs_file_extended_cube']), decode_times=False)
gpps = mod_dat_f[obs_dict['gpp']['obs_var']].values.reshape(-1, 360, 720)
mod_dat_f.close()

mod_dat_f = xr.open_dataset(os.path.join(top_dir, obs_dict['c_veg']['obs_file_extended_cube']), decode_times=False)
c_vegs = mod_dat_f[obs_dict['c_veg']['obs_var']].values.reshape(-1, 360, 720)
mod_dat_f.close()

mod_dat_f = xr.open_dataset(os.path.join(top_dir, obs_dict['c_soil']['obs_file_extended_cube']), decode_times=False)
c_soils = mod_dat_f['c_soil'].values.reshape(-1, 360, 720)
mod_dat_f.close()

#--------------------------

# plt.figure(figsize=(8,12))
# cSI = 1
# for _cS in c_soils:
#     _ax = plt.subplot(len(c_soils),
#                         1,
#                         cSI,
#                         projection=ccrs.Robinson(central_longitude=0),
#                         frameon=False)
#     make_map(_ax, 'c_soil', _cS)
#     cSI += 1
# plt.savefig(co_settings['fig_settings']['fig_dir'] + 'xtra.c_soil.fig_' + fig_num + co_settings['exp_suffix'] + '.' +
#                 fig_set['fig_format'],
#                 bbox_inches='tight',
#                 dpi=fig_set['fig_dpi'])
# plt.close()
# # plt.show()
# plt.figure(figsize=(8,12))
# cSI = 1
# for _cS in c_vegs:
#     _ax = plt.subplot(len(c_vegs),
#                         1,
#                         cSI,
#                         projection=ccrs.Robinson(central_longitude=0),
#                         frameon=False)
#     make_map(_ax, 'c_veg', _cS)
#     cSI += 1
# plt.savefig(co_settings['fig_settings']['fig_dir'] + 'xtra.c_veg.fig_'+ fig_num + co_settings['exp_suffix'] + '.' +
#                 fig_set['fig_format'],
#                 bbox_inches='tight',
#                 dpi=fig_set['fig_dpi'])
# plt.close()

# # gpps=gpps[0:3]
# plt.figure(figsize=(12,8))
# cSI = 1
# for _cS in gpps:
#     _ax = plt.subplot(5,
#                         5,
#                         cSI,
#                         projection=ccrs.Robinson(central_longitude=0),
#                         frameon=False)
#     make_map(_ax, 'gpp', _cS)
#     cSI += 1
# plt.savefig(co_settings['fig_settings']['fig_dir'] + 'xtra.gpp.fig_' + fig_num + co_settings['exp_suffix'] + '.' +
#                 fig_set['fig_format'],
#                 bbox_inches='tight',
#                 dpi=fig_set['fig_dpi'])

# plt.close()
# kera
#---------------------------------------------------------------------------
# figure and plot settings
#---------------------------------------------------------------------------

# fig = plt.figure(figsize=(7, 7))
# plt.subplots_adjust(hspace=0.32, wspace=0.32)

firstTime = True
fit_method = co_settings['fig_settings']['fit_method']
data_sel = {
    'All': {
        "c_soils_sel": np.copy(c_soils),
        "c_veg_sel": np.copy(c_vegs),
        # "c_veg_sel": np.copy(c_vegs[-1]).reshape(-1, 360, 720),
        "gpp_sel": np.copy(gpps),
        "spStart": 1,
        "title": "$All$"
    },
    'c_soil': {
        "c_soils_sel": np.copy(c_soils),
        "c_veg_sel": np.copy(c_vegs[-1]).reshape(-1, 360, 720),
        "gpp_sel": np.copy(gpps[-1]).reshape(-1, 360, 720),
        "spStart": 5,
        "title": obs_dict['c_soil']['title']
    },
    'cVeg': {
        "c_soils_sel": np.copy(c_soils[-1].reshape(-1, 360, 720)),
        "c_veg_sel": np.copy(c_vegs),
        "gpp_sel": np.copy(gpps[-1]).reshape(-1, 360, 720),
        "spStart": 9,
        "title": obs_dict['c_veg']['title']
    },
    'GPP': {
        "c_soils_sel": np.copy(c_soils[-1].reshape(-1, 360, 720)),
        "c_veg_sel": np.copy(c_vegs[-1]).reshape(-1, 360, 720),
        "gpp_sel": np.copy(gpps),
        "spStart": 13,
        "title": obs_dict['gpp']['title']
    }
}

rela_tions = OrderedDict({
    'tas-tau_c': {
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
    'tas-gpp': {
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


for rel_t in rela_tions.keys():
    firstTime = True
    fig = plt.figure(figsize=(6, 7))
    plt.subplots_adjust(hspace=0.32, wspace=0.32)
    for _sel in ['All']:
        dic_ind = 0
        c_soils_sel = data_sel[_sel]['c_soils_sel']
        c_veg_sel = data_sel[_sel]['c_veg_sel']
        gpp_sel = data_sel[_sel]['gpp_sel']
        spStart = data_sel[_sel]['spStart']
        full_pred = get_full_d()
        c_soil_i = 0
        sp_index = 1
        for c_soil in c_soils_sel:
            gpp_i = 0
            for gpp in gpp_sel:
                mod_gpp = _apply_a_mask(np.copy(gpp), all_mask)
                c_veg_i = 0
                for c_veg in c_veg_sel:
                    c_total = c_soil + c_veg
                    c_total[c_total < 0.001] = np.nan
                    mod_c_total = _apply_a_mask(np.copy(c_total), all_mask)
                    mod_tau_c = mod_c_total / mod_gpp
                    mod_tau_c[mod_tau_c > 1e4] = np.nan
                    is_inf = np.isinf(mod_tau_c)
                    mod_tau_c[is_inf] = np.nan
                    mod_tau_c = _apply_a_mask(mod_tau_c, all_mask)
                    mod_tas = np.copy(mod_tas_0)
                    mod_pr = np.copy(mod_pr_0)
                    mod_pr, mod_gpp, mod_arI, mod_tas, mod_tau_c = _apply_common_mask_g(
                        mod_pr, mod_gpp, arI, mod_tas, mod_tau_c)

                    print('------------------------------------------------------')
                    print('c_soil_i:', c_soil_i, 'c_veg_i:', c_veg_i, 'gpp_i:', gpp_i)
                    print('------------------------------------------------------')
                    x_var = rel_t.split('-')[0]
                    y_var = rel_t.split('-')[1]
                    plt.subplot(4, 3, spStart + sp_index - 1)
                    ptool.ax_orig(axfs=ax_fs * 0.9)

                    dat_x_var = vars()['mod_' + x_var]
                    dat_y_var = vars()['mod_' + y_var]

                    pred_dic = get_fitted_data(dat_x_var,
                                dat_y_var,
                                _logY=rela_tions[rel_t]['log_y'],
                                intercept=rela_tions[rel_t]['inter_cept'],
                                _fit_method=fit_method,
                                bounds=rela_tions[rel_t]['p_bounds'])
                    # full_pred = fill_pred_d(full_pred, pred_dic, sp_index)
                    h = plt.text(0.02,
                                0.97,
                                al[spStart + sp_index - 2],
                                weight='bold',
                                fontsize=ax_fs,
                                rotation=0,
                                transform=plt.gca().transAxes)
                    ptool.put_ticks(nticks=4, which_ax = 'both')
                    if sp_index == 1:
                        plt.title('GPP-RSM', ha='center', fontsize=ax_fs * 0.9)
                        plt.ylabel('Sandermann', ha='center', fontsize=ax_fs * 0.9)
                    if sp_index == 2:
                        plt.title('GPP-RS', ha='center', fontsize=ax_fs * 0.9)
                    if sp_index == 3:
                        plt.title('GPP-MTE', ha='center', fontsize=ax_fs * 0.9)
                        if firstTime:
                            for _ar in range(len(aridity_list)):
                                plt.plot(pred_dic[_ar + 1]['x'] * np.nan,
                                        pred_dic[_ar + 1]['x'] * np.nan,
                                        c=color_list[_ar],
                                        ls='-',
                                        lw=1.15,
                                        marker=None,
                                        label=aridity_list[_ar],
                                        alpha=1)
                            leg = _draw_legend_aridity(co_settings, loc_a=(-1.02422, 1.21625855))
                            firstTime = False
                    if sp_index == 4:
                        plt.ylabel('Soilgrids', ha='center', fontsize=ax_fs * 0.9)
                    if sp_index == 7:
                        plt.ylabel('LANDGIS', ha='center', fontsize=ax_fs * 0.9)
                    if sp_index == 10:
                        plt.ylabel('C2014', ha='center', fontsize=ax_fs * 0.9)

                    
                    # if spStart == 1:
                    #     plt.title(obs_dict[y_var]['title'] + ' (' + obs_dict[y_var]['unit'] + ')', ha='center', fontsize=ax_fs * 0.9)
                    # if sp_index == 4:
                    #     plt.ylabel(data_sel[_sel]['title'], ha='center', fontsize=ax_fs * 0.9)
                    #     plt.gca().yaxis.set_label_position("right")
                    # if spStart == 13:
                    #     plt.xlabel(obs_dict[x_var]['title'] + ' (' + obs_dict[x_var]['unit'] + ')', ha='center', fontsize=ax_fs * 0.9)
                        # ptool.rotate_labels(which_ax='x', rot=90, axfs=ax_fs)

                    if rela_tions[rel_t]['log_y']:
                        plt.gca().set_yscale('log')

                    plt.xlim(obs_dict[x_var]['plot_range_fit'][0],
                    obs_dict[x_var]['plot_range_fit'][1])
                    plt.ylim(obs_dict[y_var]['plot_range_fit'][0],
                    obs_dict[y_var]['plot_range_fit'][1])
                    c_veg_i = c_veg_i + 1

                sp_index = sp_index + 1

                gpp_i = gpp_i + 1
            c_soil_i = c_soil_i + 1
    # leg = _draw_legend_aridity(co_settings, loc_a=(-1.02422, 1.21625855))
    # firstTime = False


    plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + rel_t +'.'+
                fig_set['fig_format'],
                bbox_inches='tight',
                bbox_extra_artists=[leg],
                dpi=fig_set['fig_dpi'])

    # plt.show()
        # plt_median(full_pred, spStart, dic_ind)
