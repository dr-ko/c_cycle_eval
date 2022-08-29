import sys, os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import json
from string import ascii_letters as al

from _shared import _get_set, _apply_a_mask, _apply_common_mask_g, _rem_nan, _get_aux_data, _get_data, _draw_legend_aridity


def get_wai_data(all_mask, co_settings):
    datfile = os.path.join(
        co_settings['top_dir'],
        'Observation/FLUXCOM/total_mean_wai_1996-2016_fluxcom_rs_meteo.nc')
    datVar = 'WAI_mean'
    mod_dat_f = xr.open_dataset(datfile, decode_times=False)
    mod_dat0 = mod_dat_f[datVar].values.reshape(-1, 360, 720).mean(0)
    mod_dat = _rem_nan(mod_dat0)
    mod_dat = _apply_a_mask(mod_dat, all_mask)
    return mod_dat


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs']
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

all_mask, mod_arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tasp = _get_data('tasp',
                                          co_settings,
                                          _co_mask=all_mask)

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

mod_wai = get_wai_data(all_mask, co_settings)
mod_tasp = all_mod_dat_tasp['lpj']
mod_pr = all_mod_dat_pr['lpj']
# plt.figure()
# plt.scatter(mod_tasp, mod_wai)
# plt.show()
x_tit_tasp = obs_dict['tasp']['title'] + ' (' + obs_dict['tasp']['unit'] + ')'
x_tit_pr = obs_dict['pr']['title'] + ' (' + obs_dict['pr']['unit'] + ')'
x_tit_wai = '$WAI$' + ' (-)'

elev_a = 10
angle = -40

# Make the plot
fig = plt.figure(figsize=(6, 9))
plt.subplots_adjust(hspace=0.0, wspace=0.)
mod_wai, mod_arI, mod_tasp, mod_pr = _apply_common_mask_g(
    mod_wai, mod_arI, mod_tasp, mod_pr)
# print(mod_gpp.shape)
plt.subplot(1, 1, 1, projection='3d')
plt.gca().view_init(elev=elev_a, azim=angle)
# plt.gca().set_facecolor('k')
plt.gca().xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.gca().yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.gca().zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.gca().xaxis._axinfo["grid"]['linewidth'] = 0.4
plt.gca().yaxis._axinfo["grid"]['linewidth'] = 0.4
plt.gca().zaxis._axinfo["grid"]['linewidth'] = 0.4
# plt.gca().yaxis._axinfo["grid"]['color'] =  (1,1,1,0.4)
# plt.gca().zaxis._axinfo["grid"]['color'] =  (1,1,1,0.4)
print('------------------------------------------------------')
print('pr-wai')

plt.gca().w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
for _tr in range(len(_aridity_bounds) - 1):
    tasp1 = _aridity_bounds[_tr]
    tasp2 = _aridity_bounds[_tr + 1]
    mod_tasp_tmp = np.ma.masked_inside(mod_arI, tasp1, tasp2).mask
    _xDat_tr = mod_tasp[mod_tasp_tmp]
    _yDat_tr = mod_pr[mod_tasp_tmp]
    _zDat_tr = mod_wai[mod_tasp_tmp]
    mevery = 1
    im = plt.plot(_xDat_tr,
                  _yDat_tr,
                  _zDat_tr,
                  marker='o',
                  markevery=mevery,
                  markersize=0.6,
                  mfc='none',
                  markeredgewidth=0.3,
                  color=color_list[_tr],
                  label=aridity_list[_tr],
                  lw=0.)
# plt.ylim(0.1, 1.1)
leg = _draw_legend_aridity(co_settings, loc_a = (-15, 3000, 1.13), is_3d=True)
plt.xlim(0, 30)
plt.ylim(0, 3000)
plt.gca().set_zlabel(x_tit_wai, ha='center', fontsize=ax_fs * 0.9)
plt.ylabel(x_tit_pr, ha='center', fontsize=ax_fs * 0.9)
plt.xlabel(x_tit_tasp, ha='center', fontsize=ax_fs * 0.9)
t_x = fig.text(0.5, 0.04, '', ha='center', fontsize=ax_fs * 0.9)
t_y = fig.text(0.02,
               0.5,
               '',
               va='center',
               ma='center',
               rotation='vertical',
               fontsize=ax_fs * 0.9)
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x, t_y],
            dpi=fig_set['fig_dpi'])
plt.close()
