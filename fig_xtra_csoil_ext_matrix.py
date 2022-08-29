import _shared_plot as ptool
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy.stats as scst
import sys
from _shared import _get_set, _apply_a_mask, _apply_common_mask_g, _get_aux_data, _get_colormap_info, _get_data, _rem_nan

import cartopy.crs as ccrs
import xarray as xr
import os

def _fix_map(axis_obj):
    """
    Beautify map object.

    Clean boundaries, coast lines, and removes the outline box/circle.
    """
    # axis_obj.set_global()
    axis_obj.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    axis_obj.coastlines(linewidth=0.4, color='grey')
    plt.gca().outline_patch.set_visible(False)
    return axis_obj


def get_hex_data(_dat1, _dat2, valrange_sc):
    print(np.nanmax(_dat1),np.nanmax(_dat2),valrange_sc)
    
    _dat1 = np.ma.masked_outside(_dat1, valrange_sc[0],
                                 valrange_sc[1]).filled(np.nan)
    _dat2 = np.ma.masked_outside(_dat2, valrange_sc[0],
                                 valrange_sc[1]).filled(np.nan)
    _dat1, _dat2 = _apply_common_mask_g(_dat1, _dat2)
    _dat1mc = np.ma.masked_invalid(_dat1).compressed()
    _dat2mc = np.ma.masked_invalid(_dat2).compressed()
    return _dat1mc, _dat2mc


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
# models_only='lpj isba'.split()

models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
nmodels = len(models)
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']

all_mask, arI, area_dat = _get_aux_data(co_settings)


file_list = []
for fi_le in sorted(os.listdir(top_dir + 'Observation/all_gpp')):
    if fi_le.endswith(".nc"):
        print(os.path.join(fi_le))
        file_list = np.append(file_list, os.path.join(top_dir + 'Observation/all_gpp', fi_le))
# kera
# models = 'VPM RS RSM GOSIF FluxSat'.split()
    #  for c_soil
    # c_soils_dic = {
    #     "0": "Sandermann",
    #     "1": "Soilgrids",
    #     "2": "LANDGIS"
    # }
obs_set_name = os.environ['c_cycle_obs_set']
models = 'Sanderman SoilGrids LANDGIS HWSD+NCSCD'.split()
if 'extnsg' in obs_set_name:
    models = 'Sanderman LANDGIS HWSD+NCSCD'.split()
    # kera
if 'extnlg' in obs_set_name:
    models = 'Sanderman SoilGrids HWSD+NCSCD'.split()
    # kera
# kera
# dat = 'CERES_GPCP  CRUNCEP_crujra_v1.1  CRUNCEP_v6  CRUNCEP_v8  GSWP3  WFDEI  era5'.split()
# mlms = 'RF ANN MARS'.split()
# varbs = 'GPP_HB GPP'.split()
# models = []
# for _v in varbs:
#     for _ml in mlms:
#         for _dat in dat:
#             vname = _dat+'-'+_ml+'-'+_dat
#             models = np.append(models, vname)
# models = np.append(models, 'MTE')

# generate the figure numbers from file name
var_list = ['c_soil']
fig_num_s = sys.argv[0].split('.py')[0].split('_')[-1].split('-')
var_info = {}
for _vnum in range(len(var_list)):
    var_info[var_list[_vnum]] = fig_num_s[_vnum]

#---------------------------------------------------------------------------
mod_dat_f = xr.open_dataset(os.path.join(top_dir, obs_dict['c_soil']['obs_file_extended_cube']), decode_times=False)
gpps = mod_dat_f[obs_dict['c_soil']['obs_var']].values.reshape(-1, 360, 720)
mod_dat_f.close()

all_mod_dat = {}
mdI = 0
for _md in models:
    all_mod_dat[_md] = _apply_a_mask(gpps[mdI], all_mask)
    print(_md, np.nanmax(all_mod_dat[_md]))
    mdI = mdI + 1
    if _md == 'MTE':
        plt.figure()
        plt.imshow(all_mod_dat[_md])
        plt.colorbar()
        plt.show()
# ------------

diffMode = 'ratio'

# models = np.append(models, 'MTE')
# nmodels = len(models)
#FIGURES SETTINGS AND PARAMETER of the figure
x0 = 0.02
y0 = 1.0
# models = 'MTE RSM RS GOSIF FluxSat VPM'.split()
# models = 'MTE RSM RS GOSIF FluxSat VPM'.split()
nmodels = len(models)
wp = 1. / nmodels
hp = wp
xsp = 0.0
aspect_data = 1680. / 4320.
ysp = -0.03
xsp_sca = wp / 3 * (aspect_data)
ysp_sca = hp / 3 * (aspect_data)

wcolo = 0.25
hcolo = 0.085 * hp * nmodels / 7.
cb_off_x = wcolo
cb_off_y = 0.06158

ax_fs = fig_set['ax_fs'] * 0.7

cb_tit = 'Ratio (Column/Row)'
gppInd = 0
for plotVar, fig_num in var_info.items():
    # plotVar='tau_c'
    _bounds_dia, cm_dia, cbticks_dia, cblabels_dia = _get_colormap_info(
        plotVar, co_settings, isratio=False)
    _bounds_rat, cm_rat, cbticks_rat, cblabels_rat = _get_colormap_info(
        plotVar, co_settings, isratio=True)

    # all_mod_dat = _get_data(plotVar, co_settings, _co_mask=all_mask)
    valrange_md = obs_dict[plotVar]['plot_range_map']
    valrange_sc = obs_dict[plotVar]['plot_range_sca']
    fig = plt.figure(figsize=(9, 6))
    for colu_m in range(nmodels):
        colu_mod = models[colu_m]
        mod_dat_colu = all_mod_dat[colu_mod]
        for row_m in range(nmodels):
            row_mod = models[row_m]
            mod_dat_row = all_mod_dat[row_mod]

            print('---- the model ' + colu_mod + ' is done for variable ' +
                row_mod + '---------')
            if colu_m == row_m:
                _ax = plt.axes([
                    x0 + colu_m * wp + colu_m * xsp, y0 -
                    (row_m * hp + row_m * ysp), wp, hp
                ],
                            projection=ccrs.Robinson(central_longitude=0),
                            frameon=False)  #,sharex=right,sharey=all)
                _fix_map(_ax)
                plot_dat = mod_dat_colu
                varName = obs_dict[plotVar]['title']
                varUnit = obs_dict[plotVar]['unit']

                plt.imshow(np.ma.masked_less(plot_dat[0:300, :], -999.),
                        interpolation='none',
                        norm=matplotlib.colors.BoundaryNorm(
                            _bounds_dia,
                            len(_bounds_dia) - 1),
                        cmap=cm_dia,
                        origin='upper',
                        transform=ccrs.PlateCarree(),
                        extent=[-180, 180, -60, 90])
            if colu_m < row_m:
                _ax = plt.axes([
                    x0 + colu_m * wp + colu_m * xsp + xsp_sca,
                    y0 - (row_m * hp + row_m * ysp) + ysp_sca,
                    wp * aspect_data, hp * aspect_data
                ])
                xdat, ydat = mod_dat_row, mod_dat_colu
                dat1h, dat2h = get_hex_data(xdat, ydat, valrange_sc)
                print(dat1h, dat2h)
                _ax.hexbin(dat1h,
                        dat2h,
                        bins='log',
                        mincnt=10,
                        gridsize=40,
                        cmap='viridis_r',
                        linewidths=0)
                plt.ylim(valrange_sc[0], valrange_sc[1] * 1.05)
                plt.xlim(valrange_sc[0], valrange_sc[1] * 1.05)
                ymin, ymax = plt.ylim()
                xmin, xmax = plt.xlim()
                plt.plot((xmin, xmax), (ymin, ymax), 'k', lw=0.1)

                r, p = scst.pearsonr(dat1h, dat2h)
                rho, p = scst.spearmanr(dat1h, dat2h)
                tit_str = "$r$=" + str(round(r, 2)) + ", $\\rho$=" + str(
                    round(rho, 2))
                plt.title(tit_str,
                        fontsize=ax_fs * 0.953,
                        ma='left',
                        y=1.175,
                        va="top")
                print(tit_str)
                if colu_m != 0 and row_m != nmodels - 1:
                    ptool.ax_clr(axfs=ax_fs)
                    ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                elif colu_m == 0 and row_m != nmodels - 1:
                    ptool.ax_clrX(axfs=ax_fs)
                    ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                elif row_m == nmodels - 1 and colu_m != 0:
                    ptool.ax_clrY(axfs=ax_fs)
                    ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                if colu_m == 0 and row_m == nmodels - 1:
                    ptool.ax_orig(axfs=ax_fs)
                    ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                    plt.ylabel('Column', fontsize=ax_fs)
                    plt.xlabel('Row', fontsize=ax_fs)
            if colu_m > row_m:
                _ax = plt.axes([
                    x0 + colu_m * wp + colu_m * xsp, y0 -
                    (row_m * hp + row_m * ysp), wp, hp
                ],
                            projection=ccrs.Robinson(central_longitude=0),
                            frameon=False)  #,sharex=right,sharey=all)
                _fix_map(_ax)
                if diffMode == 'ratio':
                    plot_dat = _rem_nan(mod_dat_colu / mod_dat_row)
                else:
                    plot_dat = _rem_nan(mod_dat_colu - mod_dat_row)

                plot_dat = _apply_a_mask(plot_dat, all_mask)
                plt.imshow(np.ma.masked_equal(plot_dat[0:300, :], -9999.),
                        norm=matplotlib.colors.BoundaryNorm(
                            _bounds_rat, len(_bounds_rat)),
                        interpolation='none',
                        cmap=cm_rat,
                        origin='upper',
                        transform=ccrs.PlateCarree(),
                        extent=[-180, 180, -60, 90])

            if colu_m == nmodels - 1 or colu_m == 0:
                if row_mod == 'obs':
                    _title_sp = 'Obs-based'
                    # _title_sp = obs_dict[plotVar]['obs_src']
                else:
                    _title_sp = row_mod
                    # _title_sp = model_dict[row_mod]['model_name']
                plt.ylabel(_title_sp, fontsize=0.809 * ax_fs)
                if colu_m == nmodels - 1:
                    plt.gca().yaxis.set_label_position("right")
            if row_m == 0 or row_m == nmodels - 1:
                if colu_mod == 'obs':
                    _title_sp = 'Obs-based'
                else:
                    _title_sp = colu_mod
                    # _title_sp = model_dict[colu_mod]['model_name']
                if row_m == 0:
                    plt.title(_title_sp, fontsize=0.809 * ax_fs)
                else:
                    plt.xlabel(_title_sp, fontsize=0.809 * ax_fs)

    t_x = plt.figtext(0.5, 0.5, ' ', transform=plt.gca().transAxes)
    x_colo = ((x0 + x0 + colu_m * wp + colu_m * xsp + wp) / 2) - cb_off_x
    x_colo = 0.02
    y_colo = y0 + hp + cb_off_y
    _axcol_dia = [x_colo, y_colo, wcolo, hcolo]
    print(x_colo)
    cb_tit_d = obs_dict[plotVar]['title'] + ' (' + obs_dict[plotVar][
        'unit'] + ')'
    cb = ptool.mk_colo_tau_c(_axcol_dia,
                        _bounds_dia,
                        cm_dia,
                        tick_locs=cbticks_dia,
                        cbfs=0.96 * ax_fs,
                        cbtitle=cb_tit_d,
                        cbrt=90)

    x_colo = ((x0 + x0 + colu_m * wp + colu_m * xsp) / 2) + cb_off_x
    x_colo = 0.76
    y_colo = y0 + hp + cb_off_y
    _axcol_rat = [x_colo, y_colo, wcolo, hcolo]
    cb = ptool.mk_colo_cont(_axcol_rat,
                            _bounds_rat,
                            cm_rat,
                            cbfs=0.78 * ax_fs,
                            cbrt=90,
                            col_scale='log',
                            cbtitle=cb_tit,
                            tick_locs=cbticks_rat)
    cb.ax.set_xticklabels(cblabels_rat,
                        fontsize=0.86 * ax_fs,
                        ha='center',
                        rotation=0)
    plt.savefig(co_settings['fig_settings']['fig_dir'] + 'ztet_'+plotVar+'_' + fig_num + co_settings['exp_suffix'] + '.' +
                fig_set['fig_format'],
                bbox_inches='tight',
                bbox_extra_artists=[t_x],
                dpi=fig_set['fig_dpi'])

