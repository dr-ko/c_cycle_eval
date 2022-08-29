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


def plt_subplot(_xDat, _yDat, _zDat):

    plt.gca().w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    for _tr in range(len(_aridity_bounds) - 1):
        tas1 = _aridity_bounds[_tr]
        tas2 = _aridity_bounds[_tr + 1]
        mod_tas_tmp = np.ma.masked_inside(mod_arI, tas1, tas2).mask
        _yDat_tr_0 = _yDat[mod_tas_tmp]
        _xDat_tr_0 = _xDat[mod_tas_tmp]
        _zDat_tr_0 = _zDat[mod_tas_tmp]
        _xDat_tr = _xDat_tr_0
        _yDat_tr = _yDat_tr_0
        _zDat_tr = _zDat_tr_0
        mevery = 50
        im = plt.plot(_xDat_tr,
                      _yDat_tr,
                      _zDat_tr,
                      marker='+',
                      markevery=mevery,
                      markersize=2,
                      mfc='none',
                      markeredgewidth=0.3,
                      color=color_list[_tr],
                      lw=0.)

    return (im)


def get_dat_bins(_variable, _dat, _tas, _pr, _area_dat):
    # get_dat_bins(mod_wai,mod_pr,mod_tas,area_dat)
    _datBins = np.ones((nBins, nBins)) * np.nan
    if _variable in co_settings['pgc_vars']:
        _dat = _dat * _area_dat

    for ii in range(nBins - 1):
        t1 = tasBins[ii]
        t2 = tasBins[ii + 1]
        tasMask = np.ma.getmask(
            np.ma.masked_invalid(
                np.ma.masked_outside(_tas, t1, t2).filled(np.nan)))
        for jj in range(nBins - 1):
            pr1 = prBins[jj]
            pr2 = prBins[jj + 1]
            prMask = np.ma.getmask(
                np.ma.masked_invalid(
                    np.ma.masked_outside(_pr, pr1, pr2).filled(np.nan)))
            _valMaskA = (1 - tasMask) * (1 - prMask)
            if np.nansum(_valMaskA) > minPoints:

                _valMask = np.ma.nonzero(_valMaskA)

                _datVal = _dat[_valMask]
                _areaVal = _area_dat[_valMask]
                if _variable in co_settings['pgc_vars']:
                    _datBins[ii, jj] = np.nansum(_datVal) / np.nansum(_areaVal)
                else:
                    _datBins[ii, jj] = np.nanmedian(_datVal)
            else:
                _datBins[ii, jj] = np.nan
    _datBins = _datBins.T
    # print('total',plotVar,np.nansum(_datBins),np.nansum(_dat)*1e-12)
    print(_datBins)
    _datBins[_datBins > 1e3] = np.nan

    return _datBins


def get_tau_c_bins(_variable, _datgpp, _datc_total, _tas, _pr, _area_dat):
    _datBins = np.ones((nBins, nBins)) * np.nan
    if _variable in co_settings['pgc_vars']:
        _datgpp = _datgpp * _area_dat
        _datc_total = _datc_total * _area_dat
    for ii in range(nBins - 1):
        t1 = tasBins[ii]
        t2 = tasBins[ii + 1]
        tasMask = np.ma.getmask(
            np.ma.masked_invalid(
                np.ma.masked_outside(_tas, t1, t2).filled(np.nan)))
        for jj in range(nBins - 1):
            pr1 = prBins[jj]
            pr2 = prBins[jj + 1]
            prMask = np.ma.getmask(
                np.ma.masked_invalid(
                    np.ma.masked_outside(_pr, pr1, pr2).filled(np.nan)))
            _valMaskA = (1 - tasMask) * (1 - prMask)
            if np.nansum(_valMaskA) > minPoints:
                _valMask = np.ma.nonzero(_valMaskA)
                _datgppVal = _datgpp[_valMask]
                _datc_totalVal = _datc_total[_valMask]
                # _areaVal=_area_dat[_valMask]
                _dattmp = np.nansum(_datc_totalVal) / np.nansum(_datgppVal)

                if _dattmp > 1.e5:
                    _datBins[ii, jj] = np.nan
                else:
                    _datBins[ii, jj] = _dattmp
            else:
                _datBins[ii, jj] = np.nan
    _datBins = _datBins.T
    _datBins[_datBins > np.nanpercentile(_datBins, 98)] = np.nan
    # print(_datBins)
    return _datBins


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

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas', co_settings, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask)
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=all_mask)
all_mod_dat_c_total = _get_data('tau_c', co_settings, _co_mask=all_mask)

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

mod_wai = get_wai_data(all_mask, co_settings)
mod_tas = all_mod_dat_tas['lpj']
mod_pr = all_mod_dat_pr['lpj']
# plt.figure()
# plt.scatter(mod_tas, mod_wai)
# plt.show()
x_tit_tas = obs_dict['tas']['title'] + ' (' + obs_dict['tas']['unit'] + ')'
x_tit_pr = obs_dict['pr']['title'] + ' (' + obs_dict['pr']['unit'] + ')'
x_tit_gpp = obs_dict['gpp']['title'] + ' (' + obs_dict['gpp']['unit'] + ')'
x_tit_tau_c = obs_dict['tau_c']['title'] + ' (' + obs_dict['tau_c']['unit'] + ')'
x_tit_wai = '$WAI$' + ' (-)'

tauMin = 5
tauMax = 1000
nBins = 50
prBins=np.linspace(0,3000,nBins)
tasBins=np.linspace(-20,30,nBins)
minPoints = 10

# #GPP_TAIR-GPP_TAIR-GPP_TAIR-GPP_TAIR-GPP_TAIR-GPP_TAIR-GPP_TAIR-GPP_TAIR-
models = 'obs'.split()
nmodels = len(models)

### PR-WAI, TAS-WAI
imgI = 1
elev_a = 0
d3_dir = os.path.join(co_settings['fig_settings']['fig_dir'], '3d/')
os.makedirs(d3_dir, exist_ok=True)

for row_m in range(nmodels):
    row_mod = models[row_m]

    mod_c_total = all_mod_dat_c_total[row_mod]
    mod_gpp = all_mod_dat_gpp[row_mod]
    mod_tau_c = all_mod_dat_tau_c[row_mod]

    mod_wai, mod_gpp, mod_c_total, mod_arI, mod_tas, mod_pr, mod_tau_c = _apply_common_mask_g(
        mod_wai, mod_gpp, mod_c_total, arI, mod_tas, mod_pr, mod_tau_c)
    print(mod_tas.mean(), mod_wai.mean(), mod_gpp.mean(), mod_tau_c.mean())
    # print(mod_gpp.shape)
    X, Y = np.meshgrid(tasBins, prBins)
    Z_wai = get_dat_bins('wai', mod_wai, mod_tas, mod_pr,
                        area_dat)  #get zonal variable correlation
    Z_gpp = get_dat_bins('gpp', mod_gpp, mod_tas, mod_pr,
                            area_dat)  #get zonal variable correlation
    Z_tau_c = get_tau_c_bins('tau_c', mod_gpp, mod_c_total, mod_tas, mod_pr,
                        area_dat)  #get zonal variable correlation
    for angle in range(-90, 0, 3):

        # Make the plot
        fig = plt.figure(figsize=(6, 9))
        plt.subplots_adjust(hspace=0.0, wspace=0.)
        plt.subplot(3, 1, 1, projection='3d')
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
        surf = plt.gca().plot_surface(X,
                                        Y,
                                        np.ma.masked_invalid(Z_wai),
                                        color='#F1B5F3',
                                        linewidth=0,
                                        antialiased=False,
                                        alpha=0.3)

        # plt.show()
        im = plt_subplot(mod_tas,
                            mod_pr,
                            mod_wai)
        plt.ylim(0.1, 1.1)
        leg = _draw_legend_aridity(co_settings, loc_a = (-15, 3000, 0.3), is_3d=True)
        plt.xlim(-20, 30)
        plt.ylim(0, 3000)
        # plt.zlabel(x_tit_wai, ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel(x_tit_pr, ha='center', fontsize=ax_fs * 0.9)
        plt.xlabel(x_tit_tas, ha='center', fontsize=ax_fs * 0.9)
        plt.gca().set_zlabel('$WAI\ (-)$', ha='center', fontsize=ax_fs * 0.9)
        h = plt.title('a',
                        weight='bold',
                        x=0.061,
                        y=0.75,
                        fontsize=ax_fs,
                        rotation=0)
        plt.subplot(3, 1, 2, projection='3d')
        print('------------------------------------------------------')
        plt.gca().xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().xaxis._axinfo["grid"]['linewidth'] = 0.4
        plt.gca().yaxis._axinfo["grid"]['linewidth'] = 0.4
        plt.gca().zaxis._axinfo["grid"]['linewidth'] = 0.4
        print('pr-wai')
        plt.gca().view_init(elev=elev_a, azim=angle)
        surf = plt.gca().plot_surface(X,
                                        Y,
                                        np.ma.masked_invalid(Z_gpp),
                                        color='#F1B5F3',
                                        linewidth=0,
                                        antialiased=False,
                                        alpha=0.3)

        plt_subplot(mod_tas,
                    mod_pr,
                    mod_gpp)
        plt.xlim(-20, 30)
        plt.ylim(0, 3000)
        plt.ylabel(x_tit_pr, ha='center', fontsize=ax_fs * 0.9)
        plt.xlabel(x_tit_tas, ha='center', fontsize=ax_fs * 0.9)
        plt.gca().set_zlabel('$GPP\ (kg/m^2/yr)$',
                                ha='center',
                                fontsize=ax_fs * 0.9)
        # plt.ylabel(y_tit_tau_c, ha='center', fontsize=ax_fs * 0.9)
        h = plt.title('b',
                        weight='bold',
                        x=0.061,
                        y=0.75,
                        fontsize=ax_fs,
                        rotation=0)
        plt.subplot(3, 1, 3, projection='3d')
        print('------------------------------------------------------')
        print('pr-wai')
        plt.gca().xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        plt.gca().xaxis._axinfo["grid"]['linewidth'] = 0.4
        plt.gca().yaxis._axinfo["grid"]['linewidth'] = 0.4
        plt.gca().zaxis._axinfo["grid"]['linewidth'] = 0.4
        plt.gca().view_init(elev=elev_a, azim=angle)
        surf = plt.gca().plot_surface(X,
                                        Y,
                                        np.ma.masked_invalid(Z_tau_c),
                                        color='#F1B5F3',
                                        linewidth=0,
                                        antialiased=False,
                                        alpha=0.3)
        plt_subplot(mod_tas,
                    mod_pr,
                    mod_tau_c)
        plt.ylim(0, 3000)
        plt.gca().set_zlim(1, 100)
        plt.gca().set_zscale('log')
        # plt.gca(). lim(1, 1000)
        plt.xlim(-20, 30)
        plt.ylim(0, 3000)
        plt.ylabel(x_tit_pr, ha='center', fontsize=ax_fs * 0.9)
        plt.xlabel(x_tit_tas, ha='center', fontsize=ax_fs * 0.9)
        zl = plt.gca().set_zlabel('$\\tau\ (years)$',
                                    ha='center',
                                    fontsize=ax_fs * 0.9)
        h = plt.title('c',
                        weight='bold',
                        x=0.061,
                        y=0.75,
                        fontsize=ax_fs,
                        rotation=0)
        t_x = fig.text(0.5, 0.04, '', ha='center', fontsize=ax_fs * 0.9)
        t_y = fig.text(0.02,
                        0.5,
                        '',
                        va='center',
                        ma='center',
                        rotation='vertical',
                        fontsize=ax_fs * 0.9)
        plt.savefig(d3_dir + 'xtra_fig_.' + fig_num +
                    format(imgI, '02') + '.png',
                    bbox_inches='tight',
                    bbox_extra_artists=[t_x, t_y, zl],
                    dpi=200)
imgI = imgI + 1
plt.close()
