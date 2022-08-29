import glob, os
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from _shared import _get_set, _apply_a_mask, _apply_common_mask_g, _get_aux_data, _get_colormap_info, _get_data, _rem_nan
import _shared_plot as ptool

import cartopy.crs as ccrs
import matplotlib.colors


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


def get_ratio_colormap(_whichVar, _co_settings, isratio=False, isdiff=False):
    import matplotlib as mpl
    obs_dict = _co_settings['obs_dict']
    if isratio:
        border = 0.1
        ncolo = 128
        num_gr = int(ncolo // 4)
        num_col = num_gr - 4

        _bounds_rat = np.concatenate(
            (np.geomspace(0.02, 0.025,
                          num=num_col), np.geomspace(0.025, 0.033, num=num_col),
             np.geomspace(0.033, 0.05,
                          num=num_col), np.geomspace(0.05, border, num=num_col),
             np.linspace(border, 1 / border,
                         num=num_gr), np.geomspace(1 / border, 2, num=num_col),
             np.geomspace(20, 30, num=num_col), np.geomspace(30, 40, num=num_col),
             np.geomspace(40, 50, num=num_col)))

        cb_ticks = [0.02, 0.025, 0.033, 0.05, 0.1, 10, 20, 30, 40, 50]

        cb_labels = [
            '  $\\dfrac{1}{50}$', '  $\\dfrac{1}{40}$', '  $\\dfrac{1}{30}$',
            '  $\\dfrac{1}{20}$', ' $\\dfrac{1}{10}$', ' $10$', ' $20$',
            ' $30$', ' $40$', ' $50$'
        ]

        # combine them and build a new colormap
        colors1 = plt.cm.Blues(np.linspace(0.15, 0.998, (num_col) * 4))[::-1]
        colorsgr = np.tile(np.array([0.8, 0.8, 0.8, 1]),
                           num_gr).reshape(num_gr, -1)
        colors2 = plt.cm.Reds(np.linspace(0.15, 0.998, (num_col) * 4))

        # combine them and build a new colormap
        colors1g = np.vstack((colors1, colorsgr))
        colors = np.vstack((colors1g, colors2))
        cm_rat_c = mpl.colors.LinearSegmentedColormap.from_list(
            'my_colormap', colors)
        norm = mpl.colors.BoundaryNorm(boundaries=_bounds_rat,
                                       ncolors=len(_bounds_rat))

        col_map = mpl.colors.LinearSegmentedColormap.from_list(
            'my_colormap', colors)
        bo_unds = _bounds_rat

    return bo_unds, col_map, cb_ticks, cb_labels

overWriteData = False
co_settings = _get_set()
plotVar = 'gpp'
_bounds_dia, cm_dia, cbticks_dia, cblabels_dia = _get_colormap_info(
    plotVar, co_settings, isratio=False)
    # get_ratio_colormap
_bounds_rat, cm_rat, cbticks_rat, cblabels_rat = get_ratio_colormap(
    plotVar, co_settings, isratio=True)
_bounds_rat, cm_rat, cbticks_rat, cblabels_rat = _get_colormap_info(
    plotVar, co_settings, isratio=True)
all_mask, arI, area_dat = _get_aux_data(co_settings)

# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"
# dat = 'CERES_GPCP  CRUNCEP_crujra_v1.1  CRUNCEP_v6  CRUNCEP_v8  GSWP3  WFDEI  era5'.split()
mlms = 'ANNnoPFT  GMDH_CV  KRR  MARSens  MTE  MTEM  MTE_Viterbo  RFmiss  SVM'.split()
# mlms = 'RF ANN MARS'.split()
varbs = 'GPP_HB GPP'.split()
prods = 'rs rsm'.split()
cb_tit_d = ''
ax_fs = 7
cb_tit = ''
f_list_all = []
for prod in prods:
    mainDir_t = '/home/skoirala/research/crescendo_tau/Data/Observation/FLUXCOM/fluxcom_all_{prod}_memb/long_term_mean/'.format(prod=prod)
    odir = '/home/skoirala/research/crescendo_tau/Data/Observation/all_gpp/'
    os.makedirs(odir, exist_ok=True)
    syr=2001

    eyr=2015
    nYrs = eyr - syr + 1
    mainDir = os.path.join(mainDir_t, '')
    # GPP_HB.KRR.8daily.2001.rs-ltMean.nc
    f_list = []
    for root, dirs, files in os.walk(mainDir):
        for file in files:
            # print (file)
            if file.endswith(".nc"):
                infile = os.path.join(mainDir, file)
                f_list = np.append(f_list, infile)
                f_list_all = np.append(f_list_all, infile)
                print(infile)

for prod in prods:
    mainDir_t = '/home/skoirala/research/crescendo_tau/Data/Observation/FLUXCOM/fluxcom_all_{prod}_memb/long_term_mean/'.format(prod=prod)
    odir = '/home/skoirala/research/crescendo_tau/Data/Observation/all_gpp/'
    os.makedirs(odir, exist_ok=True)
    syr=2001

    eyr=2015
    nYrs = eyr - syr + 1
    mainDir = os.path.join(mainDir_t, '')
    # GPP_HB.KRR.8daily.2001.rs-ltMean.nc
    f_list = []
    for root, dirs, files in os.walk(mainDir):
        for file in files:
            # print (file)
            if file.endswith(".nc"):
                infile = os.path.join(mainDir, file)
                f_list = np.append(f_list, infile)
                f_list_all = np.append(f_list_all, infile)
                print(infile)


    odat = np.ones((len(f_list), 360,720))
    fi_ind = 0
    for _fi in range(len(f_list)):
        fi_n = f_list[_fi]
        indat = xr.open_dataset(fi_n)
        datv = indat['GPP'].values
        odat[fi_ind]=datv
        fi_ind = fi_ind + 1
        if fi_ind < len(f_list) - 1:
            indat.close()

    ofile = os.path.join(odir, 'fluxcom_{prod}_ensemble_median.nc'.format(prod=prod))
    indat['GPP'].values = np.nanmedian(odat,0)
    indat.to_netcdf(ofile)
    print(ofile, ' ------saved')
    print('-------------------------------')
    plt.figure(figsize=(6,8))
    plt.suptitle(prod.upper(), y=0.95, fontsize=ax_fs, color='red')
    plot_dat = np.nanmedian(odat,0)
    plot_dat = _apply_a_mask(plot_dat, all_mask)
    _ax = plt.axes([0.05, 0.65, 0.9, 0.25],
                projection=ccrs.Robinson(central_longitude=0),
                frameon=False)  #,sharex=right,sharey=all)
    _fix_map(_ax)
    plt.title('Median', fontsize=ax_fs)

    plt.imshow(np.ma.masked_less(plot_dat[0:300, :], -999.),
            interpolation='none',
            norm=matplotlib.colors.BoundaryNorm(
                _bounds_dia,
                len(_bounds_dia) - 1),
            cmap=cm_dia,
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -60, 90])

    _axcol_dia = [0.25, 0.65, 0.5, 0.0088]
    cb = ptool.mk_colo_tau_c(_axcol_dia,
                        _bounds_dia,
                        cm_dia,
                        tick_locs=cbticks_dia,
                        cbfs=0.96 * ax_fs,
                        cbtitle=cb_tit_d,
                        cbrt=90)


    plot_dat = np.nanpercentile(odat, 75, axis=0) - np.nanpercentile(odat, 25, axis=0) 
    plot_dat = _apply_a_mask(plot_dat, all_mask)

    _ax = plt.axes([0.05, 0.35, 0.9, 0.25],
                projection=ccrs.Robinson(central_longitude=0),
                frameon=False)  #,sharex=right,sharey=all)
    _fix_map(_ax)
    plt.title('IQR', fontsize=ax_fs)
    plt.imshow(np.ma.masked_less(plot_dat[0:300, :], -999.),
            interpolation='none',
            norm=matplotlib.colors.BoundaryNorm(
                _bounds_dia* 0.5,
                len(_bounds_dia) - 1),
            cmap=cm_dia,
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -60, 90])

    _axcol_dia = [0.25, 0.35, 0.5, 0.0088]
    cb = ptool.mk_colo_tau_c(_axcol_dia,
                        _bounds_dia * 0.5,
                        cm_dia,
                        tick_locs=cbticks_dia* 0.5,
                        cbfs=0.96 * ax_fs,
                        cbtitle=cb_tit_d,
                        cbrt=90)

    plot_dat = (np.nanpercentile(odat, 75, axis=0) - np.nanpercentile(odat, 25, axis=0))/np.nanmedian(odat,0)
    plot_dat = _apply_a_mask(plot_dat, all_mask)
    _ax = plt.axes([0.05, 0.05, 0.9, 0.25],
                projection=ccrs.Robinson(central_longitude=0),
                frameon=False)  #,sharex=right,sharey=all)
    _fix_map(_ax)
    plt.title('IQR/Median', fontsize=ax_fs)
    plt.imshow(np.ma.masked_equal(plot_dat[0:300, :], -9999.),
            norm=matplotlib.colors.BoundaryNorm(
                _bounds_rat, len(_bounds_rat)),
            interpolation='none',
            cmap=cm_rat,
            origin='upper',
            transform=ccrs.PlateCarree(),
            extent=[-180, 180, -60, 90])


    _axcol_rat = [0.25, 0.025, 0.5, 0.0088]
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

    fig_file = os.path.join(odir, 'fluxcom_{prod}_ensemble_median.png'.format(prod=prod))
    plt.savefig(fig_file,bbox_inches='tight',dpi=300)

odat = np.ones((len(f_list_all), 360,720))
fi_ind = 0
for _fi in range(len(f_list_all)):
    fi_n = f_list_all[_fi]
    indat = xr.open_dataset(fi_n)
    datv = indat['GPP'].values
    odat[fi_ind]=datv
    print(fi_n)
    fi_ind = fi_ind + 1
    if fi_ind < len(f_list_all) - 1:
        indat.close()

plt.figure(figsize=(6,8))
plt.suptitle('ALL RS and RSM', y=0.95, fontsize=ax_fs, color='red')
plot_dat = np.nanmedian(odat,0)
plot_dat = _apply_a_mask(plot_dat, all_mask)
_ax = plt.axes([0.05, 0.65, 0.9, 0.25],
            projection=ccrs.Robinson(central_longitude=0),
            frameon=False)  #,sharex=right,sharey=all)
_fix_map(_ax)
plt.title('Median', fontsize=ax_fs)

plt.imshow(np.ma.masked_less(plot_dat[0:300, :], -999.),
        interpolation='none',
        norm=matplotlib.colors.BoundaryNorm(
            _bounds_dia,
            len(_bounds_dia) - 1),
        cmap=cm_dia,
        origin='upper',
        transform=ccrs.PlateCarree(),
        extent=[-180, 180, -60, 90])

_axcol_dia = [0.25, 0.65, 0.5, 0.0088]
cb = ptool.mk_colo_tau_c(_axcol_dia,
                    _bounds_dia,
                    cm_dia,
                    tick_locs=cbticks_dia,
                    cbfs=0.96 * ax_fs,
                    cbtitle=cb_tit_d,
                    cbrt=90)


plot_dat = np.nanpercentile(odat, 75, axis=0) - np.nanpercentile(odat, 25, axis=0)
plot_dat = _apply_a_mask(plot_dat, all_mask)

_ax = plt.axes([0.05, 0.35, 0.9, 0.25],
            projection=ccrs.Robinson(central_longitude=0),
            frameon=False)  #,sharex=right,sharey=all)
_fix_map(_ax)
plt.title('IQR', fontsize=ax_fs)
plt.imshow(np.ma.masked_less(plot_dat[0:300, :], -999.),
        interpolation='none',
        norm=matplotlib.colors.BoundaryNorm(
            _bounds_dia* 0.5,
            len(_bounds_dia) - 1),
        cmap=cm_dia,
        origin='upper',
        transform=ccrs.PlateCarree(),
        extent=[-180, 180, -60, 90])

_axcol_dia = [0.25, 0.35, 0.5, 0.0088]
cb = ptool.mk_colo_tau_c(_axcol_dia,
                    _bounds_dia * 0.5,
                    cm_dia,
                    tick_locs=cbticks_dia* 0.5,
                    cbfs=0.96 * ax_fs,
                    cbtitle=cb_tit_d,
                    cbrt=90)

plot_dat = (np.nanpercentile(odat, 75, axis=0) - np.nanpercentile(odat, 25, axis=0))/np.nanmedian(odat,0)
plot_dat = _apply_a_mask(plot_dat, all_mask)
_ax = plt.axes([0.05, 0.05, 0.9, 0.25],
            projection=ccrs.Robinson(central_longitude=0),
            frameon=False)  #,sharex=right,sharey=all)
_fix_map(_ax)
plt.title('IQR/Median', fontsize=ax_fs)
plt.imshow(np.ma.masked_equal(plot_dat[0:300, :], -9999.),
        norm=matplotlib.colors.BoundaryNorm(
            _bounds_rat, len(_bounds_rat)),
        interpolation='none',
        cmap=cm_rat,
        origin='upper',
        transform=ccrs.PlateCarree(),
        extent=[-180, 180, -60, 90])


_axcol_rat = [0.25, 0.025, 0.5, 0.0088]
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

fig_file = os.path.join(odir, 'fluxcom_all_ensemble_median.png'.format(prod=prod))
plt.savefig(fig_file,bbox_inches='tight',dpi=300)
