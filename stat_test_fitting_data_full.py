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
import pandas as pd
import pingouin as pg
#--------
import pickle
import seaborn as sns


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
#---------------------------------------------------------------------------
# get the observations (Carvalhais, 2014) and add them to the full cube
#---------------------------------------------------------------------------
co_settings['model']['names'] = models


# fig = plt.figure(figsize=(7, 7))
# plt.subplots_adjust(hspace=0.32, wspace=0.32)

firstTime = True
fit_method = co_settings['fig_settings']['fit_method']
# data_sel = {
#     'All': {
#         "c_soils_sel": np.copy(c_soils),
#         "c_veg_sel": np.copy(c_vegs),
#         "gpp_sel": np.copy(gpps),
#         "spStart": 1,
#         "title": "$All$"
#     }
    # ,
    # 'c_soil': {
    #     "c_soils_sel": np.copy(c_soils),
    #     "c_veg_sel": np.copy(c_vegs[-1]).reshape(-1, 360, 720),
    #     "gpp_sel": np.copy(gpps[-1]).reshape(-1, 360, 720),
    #     "spStart": 5,
    #     "title": obs_dict['c_soil']['title']
    # },
    # 'cVeg': {
    #     "c_soils_sel": np.copy(c_soils[-1].reshape(-1, 360, 720)),
    #     "c_veg_sel": np.copy(c_vegs),
    #     "gpp_sel": np.copy(gpps[-1]).reshape(-1, 360, 720),
    #     "spStart": 9,
    #     "title": obs_dict['c_veg']['title']
    # },
    # 'GPP': {
    #     "c_soils_sel": np.copy(c_soils[-1].reshape(-1, 360, 720)),
    #     "c_veg_sel": np.copy(c_vegs[-1]).reshape(-1, 360, 720),
    #     "gpp_sel": np.copy(gpps),
    #     "spStart": 13,
    #     "title": obs_dict['gpp']['title']
    # }
# }

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

# for _sel in data_sel.keys():
#     all_pred = {}
#     dic_ind = 0
#     c_soils_sel = data_sel[_sel]['c_soils_sel']
#     c_veg_sel = data_sel[_sel]['c_veg_sel']
#     gpp_sel = data_sel[_sel]['gpp_sel']
#     spStart = data_sel[_sel]['spStart']
#     c_soil_i = 0
#     for c_soil in c_soils_sel:
#         all_pred[c_soil_i]={}
#         c_veg_i = 0
#         for c_veg in c_veg_sel:
#             all_pred[c_soil_i][c_veg_i]={}
#             c_total = c_soil + c_veg
#             c_total[c_total < 0.001] = np.nan
#             gpp_i = 0
#             for gpp in gpp_sel:
#                 mod_gpp = _apply_a_mask(np.copy(gpp), all_mask)
#                 mod_c_total = _apply_a_mask(np.copy(c_total), all_mask)
#                 mod_tau_c = mod_c_total / mod_gpp
#                 mod_tau_c[mod_tau_c > 1e4] = np.nan
#                 is_inf = np.isinf(mod_tau_c)
#                 mod_tau_c[is_inf] = np.nan
#                 mod_tau_c = _apply_a_mask(mod_tau_c, all_mask)
#                 mod_tas = np.copy(mod_tas_0)
#                 mod_pr = np.copy(mod_pr_0)
#                 mod_pr, mod_gpp, mod_arI, mod_tas, mod_tau_c = _apply_common_mask_g(
#                     mod_pr, mod_gpp, arI, mod_tas, mod_tau_c)

#                 sp_index = 1

#                 print('------------------------------------------------------')
#                 print('c_soil_i:', c_soil_i, 'c_veg_i:', c_veg_i, 'gpp_i:', gpp_i)
#                 print('------------------------------------------------------')
#                 for rel_t in rela_tions.keys():
#                     x_var = rel_t.split('-')[0]
#                     y_var = rel_t.split('-')[1]
#                     dat_x_var = vars()['mod_' + x_var]
#                     dat_y_var = vars()['mod_' + y_var]

#                     pred_dic = get_fitted_data(dat_x_var,
#                                 dat_y_var,
#                                 _logY=rela_tions[rel_t]['log_y'],
#                                 intercept=rela_tions[rel_t]['inter_cept'],
#                                 _fit_method=fit_method,
#                                 bounds=rela_tions[rel_t]['p_bounds'])
#                     all_pred[c_soil_i][c_veg_i][gpp_i]=pred_dic

#                 gpp_i = gpp_i + 1
#                 dic_ind = dic_ind + 1
#             c_veg_i = c_veg_i + 1
#         c_soil_i = c_soil_i + 1


with open(co_settings['fig_settings']['fig_dir'] + 'all_fitted_data' + co_settings['exp_suffix'] + '.pkl', 'rb') as f:
    all_pred = pickle.load(f)
rel_n = 'tas-tau_c'
head = ['c_soil', 'c_veg', 'gpp', '1_a','1_b','1_c','2_a','2_b','2_c','3_a','3_b','3_c','4_a','4_b','4_c']
flat_dic = {}
for _hd in head:
    flat_dic[_hd] = []
# all_pred_df = pd.data
# print (all_pred.keys(),all_pred['c_soil_1'].keys(),all_pred['c_soil_1']['c_veg_1'].keys(),all_pred['c_soil_1']['c_veg_1']['gpp_1'].keys(),all_pred['c_soil_1']['c_veg_1']['gpp_1'][rel_n][1].keys()) #,all_pred[0][0][0][1]['coef'])
for csoil in range(1,5):
    for cveg in range(1,5):
        for gpp in range(1,4):
            flat_dic['c_soil']=np.append(flat_dic['c_soil'], csoil)
            flat_dic['c_veg']=np.append(flat_dic['c_veg'], cveg)
            flat_dic['gpp']=np.append(flat_dic['gpp'], gpp)
            dat = all_pred['c_soil_'+str(csoil)]['c_veg_'+str(cveg)]['gpp_'+str(gpp)][rel_n]
            for clim in dat.keys():
                coef = dat[clim]['coef']
                flat_dic[str(clim)+'_a'] = np.append(flat_dic[str(clim)+'_a'], coef[0])
                flat_dic[str(clim)+'_b'] = np.append(flat_dic[str(clim)+'_b'], coef[1])
                flat_dic[str(clim)+'_c'] = np.append(flat_dic[str(clim)+'_c'], coef[2])
# print(flat_dic['c_soil'])
dat_f = pd.DataFrame.from_dict(flat_dic)
dat_f['offset'] = dat_f['1_c']-dat_f['4_c']
params ='a b c offset'.split()
clims = '1 2 3 4'.split()

# histogram
plt.figure()
plt.suptitle('All cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            datplot = dat_f['1_c'] - dat_f[clim+'_c']
            if clim == '1':
                climn = climn + 1
                continue
        else:
            datplot = dat_f[clim+'_'+param]
        plt.hist(datplot, bins=50, density=False, color=color_list[climn])
        climn = climn + 1
    plt.title(param)
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_hist_all_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])
# kde
plt.figure()
plt.suptitle('All cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            datplot = dat_f['1_c'] - dat_f[clim+'_c']
        else:
            datplot = dat_f[clim+'_'+param]
        sns.kdeplot(datplot,
                 label = aridity_list[climn],color=color_list[climn])
        # plt.hist(dat_f[clim+'_'+param], bins=50, density=True, color=color_list[climn])
        climn = climn + 1
    plt.title(param)
    plt.xlabel('')
    plt.ylabel('')
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_kde_all_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])

# box plot
plt.figure()
plt.suptitle('All cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            dat_f['offset'] = dat_f['1_c'] - dat_f[clim+'_c']
            # dat_f['offset'] = 10**dat_f['1_c'] - 10**dat_f[clim+'_c']
            if clim == '1':
                dat_f['offset'] = dat_f['offset'] * np.nan
            y_var = 'offset'
        else:
            datplot = dat_f[clim+'_'+param]
            y_var = clim+'_'+param
        bp=plt.boxplot(dat_f[y_var], positions=[climn],showfliers=False, notch=True, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor = color_list[climn], alpha=0.8)
            box.set(linewidth=0)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element],color = color_list[climn] )
        climn = climn + 1
    plt.xticks(np.arange(len(clims)),labels=aridity_list)
    plt.title(param)
    plt.xlabel('')
    plt.ylabel('')
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_box_all_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])

##################EXCLUDE LANDGIS

dat_f = dat_f[dat_f.c_soil != 3]

# histogram
plt.figure()
plt.suptitle('No LANDGIS cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            datplot = dat_f['1_c'] - dat_f[clim+'_c']
            if clim == '1':
                climn = climn + 1
                continue
        else:
            datplot = dat_f[clim+'_'+param]
        plt.hist(datplot, bins=50, density=False, color=color_list[climn])
        climn = climn + 1
    plt.title(param)
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_hist_nolgis_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])

# box
plt.figure()
plt.suptitle('No LANDGIS cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            dat_f['offset'] = dat_f['1_c'] - dat_f[clim+'_c']
            # dat_f['offset'] = 10**dat_f['1_c'] - 10**dat_f[clim+'_c']
            if clim == '1':
                dat_f['offset'] = dat_f['offset'] * np.nan
            y_var = 'offset'
        else:
            datplot = dat_f[clim+'_'+param]
            y_var = clim+'_'+param
        bp=plt.boxplot(dat_f[y_var], positions=[climn],showfliers=False, notch=True, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor = color_list[climn], alpha=0.8)
            box.set(linewidth=0)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element],color = color_list[climn] )
        climn = climn + 1
    plt.xticks(np.arange(len(clims)),labels=aridity_list)
    plt.title(param)
    plt.xlabel('')
    plt.ylabel('')
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_box_nolgis_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])



plt.figure()
plt.suptitle('No LANDGIS cSoil')
plt.subplots_adjust(hspace=0.32, wspace=0.32)
spn = 1
for param in params:
    plt.subplot(2,2,spn)
    ptool.ax_orig(axfs=ax_fs * 0.9)
    climn = 0
    for clim in clims:
        if param == 'offset':
            # datplot = 10**dat_f['1_c'] - 10**dat_f[clim+'_c']
            datplot = dat_f['1_c'] - dat_f[clim+'_c']
        else:
            datplot = dat_f[clim+'_'+param]
        sns.kdeplot(datplot,
                 label = aridity_list[climn],color=color_list[climn])
        # plt.hist(dat_f[clim+'_'+param], bins=50, density=True, color=color_list[climn])
        climn = climn + 1
    plt.title(param)
    plt.xlabel('')
    plt.ylabel('')
    spn = spn + 1
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'z_params_kde_nolgis_fig_' + fig_num + co_settings['exp_suffix'] + '.' +
            fig_set['fig_format'],
            bbox_inches='tight',
            dpi=fig_set['fig_dpi'])

print(fig_num)
# plt.show()
print(dat_f)
print('ANCOVA results')

print('climate type:','arid - humid')
print('Independent:','offset (difference of intercept)')
print('Factor: c_soil')
print('control: gpp and cveg')
aov=pg.ancova(data=dat_f, dv='offset', covar=['c_veg','gpp'], between='c_soil')
print(aov)
print('----------------------------------------------------')

print('climate type:','arid - humid')
print('Independent:','offset (difference of intercept)')
print('Factor: gpp')
print('control: c_soil and cveg')
aov=pg.ancova(data=dat_f, dv='offset', covar=['c_veg','c_soil'], between='gpp')
print(aov)
print('----------------------------------------------------')

climt={'1':'arid',
'2':'Semi-arid',
'3':'Sub-humid',
'4':'Humid'}
for pcof in ['a','b','c']:
    # pcof = 'b'
    # print('Test if slopes are different for different cSoil')
    print('#################################################')
    for clim in range(1,5):
        print('climate type:',climt[str(clim)])
        print('Independent:',pcof)
        print('Factor: c_soil')
        print('control: gpp')
        aov=pg.ancova(data=dat_f, dv=str(clim)+'_'+pcof, covar=['gpp'], between='c_soil')
        print(aov)
        print('----------------------------------------------------')

for pcof in ['a','b','c']:
    # pcof = 'b'
    # print('Test if slopes are different for different cSoil')
    print('#################################################')
    for clim in range(1,5):
        print('climate type:',climt[str(clim)])
        print('Independent:',pcof)
        print('Factor: gpp')
        print('control: c_soil')
        aov=pg.ancova(data=dat_f, dv=str(clim)+'_'+pcof, covar=['c_soil'], between='gpp')
        print(aov)
        print('----------------------------------------------------')

# plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' +
#             fig_set['fig_format'],
#             bbox_inches='tight',
#             bbox_extra_artists=[leg],
#             dpi=fig_set['fig_dpi'])