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
from _shared import _get_set, _apply_a_mask, _rem_nan, _apply_common_mask_g,_get_aux_data, _get_data
#--------


import sys
import os
import os.path
import xarray as xr
import scipy.io as scio
import numpy as np

from netCDF4 import Dataset
from numpy import *



def create_nc_full(_varinfo, _dat, _lat, _lon):
    ofile_n = varInfo[1] + '_' + a_set + '_full.nc'
    _var = varInfo[2]
    _varMed = varInfo[2]

    ofile = Dataset(ofile_n, "w", format="NETCDF4")
    ofile.createDimension('lat', size(_lat))
    ofile.createDimension('lon', size(_lon))
    ofile.createDimension('level', len(_dat))
    lats = ofile.createVariable("lat", "double", ("lat", ), fill_value=np.nan)
    lats.long_name = 'latitude'
    lats.units = "degrees_north"
    lats.short_name = "lat"
    lons = ofile.createVariable("lon", "double", ("lon", ), fill_value=np.nan)
    lons.long_name = "longitude"
    lons.units = "degrees_east"
    lons.short_name = "lon"
    lons[:] = _lon
    lats[:] = _lat
    _vard = ofile.createVariable(_var,
                                 "double", ("level", "lat", "lon"),
                                 fill_value=np.nan)
    _vard.long_name = _varinfo[4][0]
    _vard.short_name = _var
    _vard.units = _varinfo[4][2]
    print(_vard, np.shape(_dat))
    _vard[:] = _rem_nan(_dat)
    ofile.institution = "Max-Planck-Institute for Biogeochemistry"
    ofile.contact = "Nuno Carvalhais [ncarval@bgc-jena.mpg.de]; Sujan Koirala [skoirala@bgc-jena.mpg.de]"
    ofile.sourceData = "various"
    ofile.citation = _varinfo[3]
    print(lats, lons, _var)
    print(ofile.data_model)
    ofile.close()
    return ()


def create_nc_sel(_varinfo, _dat, _lat, _lon):
    ofile_n = varInfo[1] + '_' + a_set + '.nc'
    _var = varInfo[2]
    _varMed = varInfo[2]
    _var5 = varInfo[2] + '_5'
    _var95 = varInfo[2] + '_95'
    ofile = Dataset(ofile_n, "w", format="NETCDF4")
    ofile.createDimension('lat', size(_lat))
    ofile.createDimension('lon', size(_lon))
    lats = ofile.createVariable("lat", "double", ("lat", ), fill_value=np.nan)
    lats.long_name = 'latitude'
    lats.units = "degrees_north"
    lats.short_name = "lat"
    lons = ofile.createVariable("lon", "double", ("lon", ), fill_value=np.nan)
    lons.long_name = "longitude"
    lons.units = "degrees_east"
    lons.short_name = "lon"
    lons[:] = _lon
    lats[:] = _lat
    # median
    _vardm = ofile.createVariable(_varMed,
                                  "double", ("lat", "lon"),
                                  fill_value=np.nan)
    _vardm.short_name = _varMed
    _vardm.long_name = _varinfo[4][0]
    _vardm.units = _varinfo[4][2]
    _vardm[:] = _rem_nan(np.nanmedian(_dat, axis=0))
    print(_vardm, np.shape(_dat))
    # 5th percentile
    _vard5 = ofile.createVariable(_var5,
                                  "double", ("lat", "lon"),
                                  fill_value=np.nan)
    _vard5.short_name = _var5
    _vard5.long_name = "fifth_percentile_" + _varinfo[4][0]
    _vard5.units = _varinfo[4][2]
    _vard5[:] = _rem_nan(np.nanpercentile(_dat, 5, axis=0))
    # 95th percentile
    _vard95 = ofile.createVariable(_var95,
                                   "double", ("lat", "lon"),
                                   fill_value=np.nan)
    _vard95.short_name = _var95
    _vard95.long_name = "ninetyfifth_percentile_" + _varinfo[4][0]
    _vard95.units = _varinfo[4][2]
    _vard95[:] = _rem_nan(np.nanpercentile(_dat, 95, axis=0))
    ofile.institution = "Max-Planck-Institute for Biogeochemistry"
    ofile.contact = "Nuno Carvalhais [ncarval@bgc-jena.mpg.de]; Sujan Koirala [skoirala@bgc-jena.mpg.de]"
    ofile.sourceData = "various"
    ofile.citation = _varinfo[3]
    print(lats, lons, _var)
    print(ofile.data_model)
    ofile.close()
    return

lats = np.linspace(-90 + 0.5 / 2, 90 - 0.5 / 2, 360)[::-1]
lons = np.linspace(-180 + 0.5 / 2, 180 - 0.5 / 2, 720)


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

# get the model data of the variable of interest
#---------------------------------------------------------------------------
# get the observations (Carvalhais, 2014) and add them to the full cube
#---------------------------------------------------------------------------
set_s = {
    # "1ma_flxc_rsm":{
    #     "carv_name": "WWW",
    #     "fan_depth": 0,
    #     "comment": "fan 1m, carvalhais 2014 1m with only HWSD"
    # },
    # "1mb_flxc_rsm":{
    #     "carv_name": "WWWandNCSCD",
    #     "fan_depth": 0,
    #     "comment": "fan 1m, carvalhais 2014 1m with HWSD and NCSCD for northern hemisphere"

    # },
    "fd_extnsg":{
        "carv_name": "FullDepth",
        "fan_depth": 2,
        "comment": "fan full, carvalhais 2014 full depth"
    }
}
for a_set,the_set in set_s.items():
    out_dir = os.path.join(top_dir + 'Observation/tau_extended_cube' + '_' + a_set +'/')
    os.makedirs(out_dir, exist_ok=True)
    gpp_c = xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/gpp_Jung2011_' + the_set['carv_name'] + '.nc'), decode_times=False)['gpp'].values
    cSoil_c = xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_soil_Carvalhais2014_' + the_set['carv_name'] + '.nc'), decode_times=False)['c_soil'].values
    c_veg_c = xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_veg_SaatchiThurner_' + the_set['carv_name'] + '.nc'), decode_times=False)['c_veg'].values

    #---------------------------------------------------------------------------
    # define the array of obs data cubes
    #---------------------------------------------------------------------------

    print(the_set['carv_name'],'fuckthisshit')
    #---------------------------------------------------------------------------
    # get the additional observations (Fan 2020)
    #---------------------------------------------------------------------------
    file_fan = h5py.File(os.path.join(
        top_dir, 'Observation/tau_database_fan2020/Component_0d5_2021.mat'),
                        mode="r")


    imin = 12
    imax = 292

    #  for c_soil
    # c_soils_dic = {
    #     "0": "Sandermann",
    #     "1": "Soilgrids",
    #     "2": "LANDGIS"
    # }

    c_soils_use = [0, 2]
    c_soils = np.ones((len(c_soils_use) + 1,360, 720)) * np.nan
    c_soil_depth = the_set['fan_depth']
    print(file_fan['Csoil'][:].shape,'fuckthisshitaswell')
    csoil_data = file_fan['Csoil'][:]
    for in_dex in range(len(c_soils_use)):
        csoil_data_depth = csoil_data[c_soil_depth]
        print(csoil_data_depth.shape, c_soils_use[in_dex])
        c_soils[in_dex][imin:imax] = csoil_data_depth[c_soils_use[in_dex]].T

    # for c_veg
    # create the cveg matrix
    # c_vegs_dic = {
    #     "0": "Saatchi",
    #     "1": "Avitabile",
    #     "2": "Saatchi-Thurner",
    #     "3": "Santoro"
    # }


    c_vegs_use = [0, 1, 2, 3]
    c_vegs_use = [0, 1, 3]
    c_vegs = np.ones((len(c_vegs_use) + 1,360, 720)) * np.nan

    for in_dex in range(len(c_vegs_use)):
        c_vegs[in_dex][imin:imax] = file_fan['Cveg'][c_vegs_use[in_dex]].T

    file_fan.close()


    #---------------------------------------------------------------------------
    # get the additional observations (FLUXCOM GPP ensembles from RS and RS + METEO)
    #---------------------------------------------------------------------------
    file_list = []
    for fi_le in sorted(os.listdir(top_dir + 'Observation/all_gpp')):
        if fi_le.endswith(".nc"):
            print(os.path.join(fi_le))
            file_list = np.append(file_list, os.path.join(top_dir + 'Observation/all_gpp', fi_le))

    # file_list = ['GPP.RS.FP-ALL.MLM-ALL.METEO-NONE.720_360.monthly.mean.2001-2015.nc', 'GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-ALL.720_360.monthly.2001-2010.ltMean.nc']
    gpps = np.ones((len(file_list), 360, 720)) * np.nan
    gcm2d_to_kgcm2yr = 365.5/1000.0
    for _i in range(len(gpps)):
        gpps[_i]=xr.open_dataset(file_list[_i])['GPP'].values #* gcm2d_to_kgcm2yr

    #---------------------------------------------------------------------------
    # put the carvalhais 2014 estimate in the cube
    #---------------------------------------------------------------------------

    # gpps[-1] = gpp_c
    c_soils[-1] = cSoil_c
    c_vegs[-1] = c_veg_c
    #---------------------------------------------------------------------------
    # remove nan and infs in the data
    #---------------------------------------------------------------------------
    gpps[gpps < 0.001] = np.nan
    is_nan = np.isnan(c_soils)
    c_soils[is_nan] = 0
    is_nan = np.isinf(c_soils)
    c_soils[is_nan] = 0
    is_nan = np.isnan(c_vegs)
    c_vegs[is_nan] = 0
    is_nan = np.isinf(c_vegs)
    c_vegs[is_nan] = 0


    # plt.figure()
    # plt.imshow(gpp1)
    # plt.colorbar(shrink=0.6)

    # plt.figure()
    # plt.imshow(gpp2)
    # plt.colorbar(shrink=0.6)
    # plt.show()
    # gpps[0] = gpp1
    # gpps[1] = gpp2
    # kera
    # gpp2 = xr.open_dataset(top_dir + 'Observation/FLUXCOM/gpp_ensembles/GPP.RS.FP-ALL.MLM-ALL.METEO-NONE.720_360.monthly.mean.2001-2015.nc')['GPP'].values

    # /home/skoirala/research/crescendo_tau/Data/Observation/FLUXCOM/gpp_ensembles/
    # /home/skoirala/research/crescendo_tau/Data/Observation/FLUXCOM/gpp_ensembles/

    ctotals = np.ones((len(c_vegs) * len(c_soils), 360, 720))
    tau_cs = np.ones((len(c_vegs) * len(c_soils)* len(gpps), 360, 720))
    csoil_i = 0
    tau_ind = 0
    ctotal_ind = 0
    for c_soil in c_soils:
        c_veg_i = 0
        mod_c_soil = _apply_a_mask(np.copy(c_soil), all_mask)
        c_soils[csoil_i] = mod_c_soil
        for c_veg in c_vegs:
            mod_c_veg = _apply_a_mask(np.copy(c_veg), all_mask)
            c_vegs[c_veg_i] = mod_c_veg
            c_total = mod_c_soil + mod_c_veg

            # c_total[c_total < 0.001] = np.nan
            mod_c_total = _apply_a_mask(np.copy(c_total), all_mask)
            ctotals[ctotal_ind] =mod_c_total
            ctotal_ind = ctotal_ind + 1
            gpp_i = 0
            for gpp in gpps:
                mod_gpp = _apply_a_mask(np.copy(gpp), all_mask)
                tau_c_i = csoil_i + c_veg_i + gpp_i
                print('------------------------------------------------------')
                print('csoil_i:', csoil_i, 'c_veg_i:', c_veg_i, 'gpp_i:', gpp_i, 'ctotal_ind', ctotal_ind, 'tau_ind', tau_ind)
                print('------------------------------------------------------')
                mod_tau_c = mod_c_total / mod_gpp
                mod_tau_c = _apply_a_mask(mod_tau_c, all_mask)
                # plt.figure()
                # plt.imshow()
                tau_cs[tau_ind] = mod_tau_c
                tau_ind = tau_ind + 1
                gpps[gpp_i] = mod_gpp

                gpp_i = gpp_i + 1
            c_veg_i = c_veg_i + 1
        csoil_i = csoil_i + 1


    obsDic = {
        'c_veg': [
            c_vegs, out_dir + 'c_veg_extended_cube', 'c_veg',
            'Saatchi et al. (2011), https://doi.org/10.1073/pnas.1019576108 & Thurner et al. (2013), https://onlinelibrary.wiley.com/doi/full/10.1111/geb.12125',
            ['vegetation_carbon_content', 'c_veg', 'kg m-2']
        ],
        'c_soil': [
            c_soils, out_dir + 'c_soil_extended_cube', 'c_soil',
            'Carvalhais et al. (2014), https://doi.org/10.1038/nature13731',
            ['soil_carbon_content', 'c_soil', 'kg m-2']
        ],
        'gpp': [
            gpps, out_dir + 'gpp_extended_cube', 'gpp',
            'Jung et al. (2011), http://dx.doi.org/10.1029/2010JG001566',
            ['gross_primary_productivity', 'GPP', 'kg m-2 yr-1']
        ],
        'tau': [
            tau_cs, out_dir + 'tau_c_extended_cube', 'tau_c',
            'Carvalhais et al. (2014), https://doi.org/10.1038/nature13731',
            ['ecosystem_carbon_turnover_time', 'tau_c', 'yr']
        ],
        'c_total': [
            ctotals, out_dir + 'c_total_extended_cube', 'c_total',
            'Carvalhais et al. (2014), https://doi.org/10.1038/nature13731',
            [' total_ecosystem_carbon_content', 'c_total', 'kg m-2']
        ]
    }


    vars = list(obsDic.keys())
    # vars='tau'.split()
    for _var in vars:
        varInfo = obsDic[_var]
        outfile = varInfo[1]
        varName = varInfo[2]
        indat = obsDic[_var][0]
        create_nc_full(varInfo, indat, lats, lons)
        create_nc_sel(varInfo, indat, lats, lons)

        print(outfile, varName, np.shape(indat))
