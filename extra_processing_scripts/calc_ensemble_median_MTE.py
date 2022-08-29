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
# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"
# dat = 'CERES_GPCP  CRUNCEP_crujra_v1.1  CRUNCEP_v6  CRUNCEP_v8  GSWP3  WFDEI  era5'.split()

mainDir_t = '/home/skoirala/research/crescendo_tau/Data/Observation/FLUXCOM/fluxcom_all_mte_memb/long_term_mean/'
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

outdat = indat.copy(deep=True)
print(np.nanmax(odat))
odat = np.ma.masked_invalid(odat).filled(np.nan)
print(np.nanmax(odat))
odat_mean = np.nanmean(odat,0)
print(np.nanmax(odat_mean))
ofile = os.path.join(odir, 'mte_ensemble_median_2001-2010.nc')
outdat.attrs['_FillValue'] = np.nan
outdat['GPP'].values = odat_mean
outdat.to_netcdf(ofile)
print(ofile, ' ------saved')
print('-------------------------------')
