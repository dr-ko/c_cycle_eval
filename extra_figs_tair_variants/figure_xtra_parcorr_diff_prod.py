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
from _shared import _get_set, _apply_a_mask, _apply_common_mask_g,_get_aux_data, _get_data, _draw_legend_aridity, _fit_least_square, _zonal_correlation
import xarray as xr
import cartopy.crs as ccrs
#--------


def get_var_name(_var, obs_dict):
    var_name = obs_dict[_var]['title']
    if "$" in var_name:
        var_name = var_name.replace("$", "")
    return var_name


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only

fig_set = co_settings['fig_settings']

fig_set['lwMainLine'] = 1.03
fig_set['lwModLine'] = 0.45

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


corela_tions = OrderedDict({
    'tau_c-tas-pr': {
        'z_var': ['pr']
    },
    'tau_c-tas-c_soil': {
        'z_var': ['c_soil']
    },
    'tau_c-tas-pr-c_soil': {
        'z_var': ['pr','c_soil']
    }
})


zonal_set = fig_set['zonal']
zonal_set['lats'] = np.linspace(-89.75, 89.75, 360, endpoint=True)[::-1]

lats = zonal_set['lats']

fig = plt.figure(figsize=(5, 7))
plt.subplots_adjust(hspace=0.32, wspace=0.32)
plt.tick_params(labelsize=ax_fs * 0.6)
tit_x = 0.5
tit_y = 1.0


csoil_labels = ['Sandermann', 'Soilgrids','LANDGIS','C2014']
for _sel in ['All']:
    dic_ind = 0
    c_soils_sel = data_sel[_sel]['c_soils_sel']
    c_veg_sel = data_sel[_sel]['c_veg_sel']
    gpp_sel = data_sel[_sel]['gpp_sel']
    spStart = data_sel[_sel]['spStart']
    sp_index = 1
    gpp_i = 0
    for gpp in gpp_sel:
        mod_gpp = _apply_a_mask(np.copy(gpp), all_mask)
        sp_index = gpp_i * 3 + 1
        data_dict = {}
        for _rel in corela_tions.keys():
            plt.subplot(3, 3, spStart + sp_index - 1)
            plt.xlim(-1, 1)
            plt.ylim(-60, 85)
            plt.axhline(y=0, lw=0.48, color='grey')
            plt.axvline(x=0, lw=0.48, color='grey')
            ptool.rem_axLine(['top', 'right'])
            var_info = {}
            var_info['x'] = 'tau_c'
            var_info['y'] = 'tas'
            var_info['z'] = corela_tions[_rel]['z_var']
            c_soil_i = 0
            for c_soil in c_soils_sel:
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
                    data_dict['pr']=mod_pr
                    data_dict['tas']=mod_tas
                    data_dict['tau_c']=mod_tau_c
                    data_dict['c_total']=mod_c_total
                    data_dict['c_soil']=c_soil
                    data_dict['gpp']=mod_gpp
                    print('------------------------------------------------------')
                    print('c_soil_i:', c_soil_i, 'c_veg_i:', c_veg_i, 'gpp_i:', gpp_i)
                    print('------------------------------------------------------')
                    var_name_x = get_var_name(var_info['x'], obs_dict)
                    var_name_y = get_var_name(var_info['y'], obs_dict)
                    x_lab = '$r_{' + var_name_x + '-'+ var_name_y+'}$'
                    if sp_index in [1, 4, 7, 10]:
                        plt.ylabel('Latitude ($^\\circ N$)', fontsize=ax_fs, ma='center')

                    plt.gca().tick_params(labelsize=ax_fs * 0.91)

                    zcorr = _zonal_correlation(data_dict, var_info, zonal_set)
                    plt.plot(np.ma.masked_equal(zcorr, np.nan),
                        lats,
                        color=color_list[c_soil_i],
                        lw=fig_set['lwModLine']
                        )
                    c_veg_i = c_veg_i + 1


                c_soil_i = c_soil_i + 1
            if firstTime and sp_index  == 2:
                for cli in range(len(csoil_labels)):
                    plt.plot(zcorr * np.nan,
                            lats,
                            color=color_list[cli],
                            lw=fig_set['lwModLine'],
                            label=csoil_labels[cli])
                leg = _draw_legend_aridity(co_settings, loc_a=(-1.02422, 1.21625855))
                firstTime=False

            x_lab = ''
            if sp_index == 1:
                var_name_z = get_var_name(var_info['z'][0], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z+'}$'
            if sp_index == 2:
                var_name_z = get_var_name(var_info['z'][0], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z+'}$'
            if sp_index == 3:
                var_name_z1 = get_var_name(var_info['z'][0], obs_dict)
                var_name_z2 = get_var_name(var_info['z'][1], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z1+', '+ var_name_z2+'}$'

            if sp_index == 3:
                plt.ylabel('RSM', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")
            if sp_index == 6:
                plt.ylabel('RS', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")
            if sp_index == 9:
                plt.ylabel('MTE', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")

            h = plt.title(x_lab,
                            x=tit_x,
                            y=tit_y,
                            weight='bold',
                            ha='center',
                            fontsize=ax_fs * 1.1,
                            rotation=0)

            sp_index = sp_index + 1
        gpp_i = gpp_i + 1
# firstTime = False


plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + 'groupby-csoil' +'.'+
            fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])


fig = plt.figure(figsize=(5, 9))
plt.subplots_adjust(hspace=0.32, wspace=0.32)
plt.tick_params(labelsize=ax_fs * 0.6)

firstTime = True
gpp_labels = ['RSM', 'RS','MTE']
for _sel in ['All']:
    dic_ind = 0
    c_soils_sel = data_sel[_sel]['c_soils_sel']
    c_veg_sel = data_sel[_sel]['c_veg_sel']
    gpp_sel = data_sel[_sel]['gpp_sel']
    spStart = data_sel[_sel]['spStart']
    c_soil_i = 0
    sp_index = 1
    for c_soil in c_soils_sel:
        sp_index = c_soil_i * 3 + 1
        data_dict = {}
        for _rel in corela_tions.keys():
            plt.subplot(4, 3, spStart + sp_index - 1)
            plt.xlim(-1, 1)
            plt.ylim(-60, 85)
            plt.axhline(y=0, lw=0.48, color='grey')
            plt.axvline(x=0, lw=0.48, color='grey')
            ptool.rem_axLine(['top', 'right'])
            var_info = {}
            var_info['x'] = 'tau_c'
            var_info['y'] = 'tas'
            var_info['z'] = corela_tions[_rel]['z_var']
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
                    data_dict['pr']=mod_pr
                    data_dict['tas']=mod_tas
                    data_dict['tau_c']=mod_tau_c
                    data_dict['c_total']=mod_c_total
                    data_dict['c_soil']=c_soil
                    data_dict['gpp']=mod_gpp
                    print('------------------------------------------------------')
                    print('c_soil_i:', c_soil_i, 'c_veg_i:', c_veg_i, 'gpp_i:', gpp_i)
                    print('------------------------------------------------------')
                    var_name_x = get_var_name(var_info['x'], obs_dict)
                    var_name_y = get_var_name(var_info['y'], obs_dict)
                    x_lab = '$r_{' + var_name_x + '-'+ var_name_y+'}$'
                    if sp_index in [1, 4, 7, 10]:
                        plt.ylabel('Latitude ($^\\circ N$)', fontsize=ax_fs, ma='center')

                    plt.gca().tick_params(labelsize=ax_fs * 0.91)

                    zcorr = _zonal_correlation(data_dict, var_info, zonal_set)
                    plt.plot(np.ma.masked_equal(zcorr, np.nan),
                            lats,
                            color=color_list[gpp_i],
                            lw=fig_set['lwModLine'])
                    c_veg_i = c_veg_i + 1
                gpp_i = gpp_i + 1
            if firstTime and sp_index  == 3:
                for gli in range(len(gpp_labels)):
                    plt.plot(zcorr * np.nan,
                            lats,
                            color=color_list[gli],
                            lw=fig_set['lwModLine'],
                            label=gpp_labels[gli])
                leg = _draw_legend_aridity(co_settings, loc_a=(-1.02422, 1.21625855))
                firstTime=False
            x_lab = ''
            if sp_index == 1:
                var_name_z = get_var_name(var_info['z'][0], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z+'}$'
            if sp_index == 2:
                var_name_z = get_var_name(var_info['z'][0], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z+'}$'
            if sp_index == 3:
                var_name_z1 = get_var_name(var_info['z'][0], obs_dict)
                var_name_z2 = get_var_name(var_info['z'][1], obs_dict)
                x_lab = '$r_{' + var_name_x + '-'+ var_name_y + ' | '+ var_name_z1+', '+ var_name_z2+'}$'

            if sp_index == 3:
                plt.ylabel('Sandermann', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")
            if sp_index == 6:
                plt.ylabel('Soilgrids', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")
            if sp_index == 9:
                plt.ylabel('LANDGIS', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")
            if sp_index == 12:
                plt.ylabel('C2014', ha='center', fontsize=ax_fs * 0.9)
                plt.gca().yaxis.set_label_position("right")

            h = plt.title(x_lab,
                            x=tit_x,
                            y=tit_y,
                            weight='bold',
                            ha='center',
                            fontsize=ax_fs * 1.1,
                            rotation=0)


            sp_index = sp_index + 1
        c_soil_i = c_soil_i + 1


plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + 'groupby-gpp' +'.'+
            fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])

    # plt.show()
        # plt_median(full_pred, spStart, dic_ind)
