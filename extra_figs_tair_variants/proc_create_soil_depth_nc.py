import os
from numpy.core.numeric import full
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io as scio
import h5py
from _shared import _get_set, _get_data, _get_aux_data, _apply_a_mask
plt.style.use('seaborn-darkgrid')
import netCDF4 as nc
def create_nc_hlf(ofile_n,varName,_dat, d_type="double"):
    lats = np.linspace(-89.75, 89.75, 360, endpoint=True)[::-1]
    lons = np.linspace(-179.75, 179.75, 720, endpoint=True)
    ofile = nc.Dataset(ofile_n, "w", format="NETCDF4")
    ofile.createDimension('lat',np.size(lats))
    ofile.createDimension('lon',np.size(lons))
    latitudes=ofile.createVariable("lat","double",("lat",),fill_value=np.nan)
    latitudes.long_name='latitude'
    latitudes.units = "degrees_north"
    latitudes.standard_name="latitude"
    longitudes=ofile.createVariable("lon","double",("lon",),fill_value=np.nan)
    longitudes.long_name="longitude"
    longitudes.units="degrees_east"
    longitudes.standard_name="longitude"
    longitudes[:]=lons
    latitudes[:]=lats
    _vardm=ofile.createVariable(varName,d_type,("lat","lon"),fill_value=np.nan)
    _vardm[:]=_dat
    ofile.institution = "Max-Planck-Institute for Biogeochemistry" 
    ofile.contact="Sujan Koirala [skoirala@bgc-jena.mpg.de]"
    ofile.close()
    print('created:', ofile_n)
    return

def get_data_mask(all_mod_dat, fill_val = None, which_mask = None):
    all_mod_mask = which_mask.copy()
    for _mod, _dat in all_mod_dat.items():
        mod_mask = ~np.ma.masked_invalid(_dat).mask
        all_mod_mask = all_mod_mask * mod_mask 
    return all_mod_mask


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']

# get the tau data before observation is added to the list of models

# add observation to models list


masks_dir = os.path.join(top_dir, './ancillary/SoilDepth/')
os.makedirs(masks_dir, exist_ok=True)

all_mask, arI, area_dat = _get_aux_data(co_settings)

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)


imin = 12
imax = 292
full_depth = np.ones((360,720)) * np.nan
infile = os.path.join(masks_dir, 'FullSoilDepth.mat')
f = scio.loadmat(infile)
full_depth[imin:imax] = f['fd'][:]

full_depth = _apply_a_mask(full_depth, all_mask)

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

c_soils_use = [0, 1]
c_soils_use = [0, 1, 2]
c_titles=["Sandermann", "Soilgrids", "LANDGIS"]
c_soils = np.ones((len(c_soils_use),360, 720)) * np.nan
c_soil_depth = 0
plt.figure(figsize=(7,7))
plt.subplots_adjust(hspace=0.3,wspace=0.3)
for in_dex in range(len(c_soils_use)):

    c_soils_1 = np.ones((360, 720)) * np.nan
    c_soils_fd = np.ones((360, 720)) * np.nan
    print(file_fan['Csoil'][:].shape)
    c_soils[in_dex][imin:imax] = file_fan['Csoil'][:][c_soils_use[in_dex]][c_soil_depth].T
    c_soils_1[imin:imax] = file_fan['Csoil'][:][0][c_soils_use[in_dex]].T
    c_soils_fd[imin:imax] = file_fan['Csoil'][:][2][c_soils_use[in_dex]].T
    c_soils_1 = _apply_a_mask(c_soils_1, all_mask)
    c_soils_fd = _apply_a_mask(c_soils_fd, all_mask)
    # c_soils_1[imin:imax] = file_fan['Csoil'][:][c_soils_use[in_dex]][0].T
    # c_soils_fd[imin:imax] = file_fan['Csoil'][:][c_soils_use[in_dex]][2].T
    plt.subplot(2,2,in_dex + 1)
    plt.scatter((full_depth-1).flatten(), (c_soils_1-c_soils_fd).flatten(),
                     s=0.23,
                     c=arI.flatten(),
                     cmap=cm_rat,
                     norm=mpl.colors.BoundaryNorm(_aridity_bounds,
                                                  len(_aridity_bounds)),
                     linewidths=0.3218,
                     alpha=0.274)
    plt.axhline(y=0,lw=0.7,color='k')
    plt.axvline(x=0,lw=0.7,color='k')
    plt.xlabel('soil depth - 1')
    plt.ylabel("csoil$_{1m}$ - csoil$_{fd}$")
    plt.title(c_titles[in_dex])
# load and plot carvalhais
# c_soils_1 = xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_soil_Carvalhais2014_WWWandNCSCD_full.nc'), decode_times=False)['c_soil'].values[0]
# c_soils_fd = xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_soil_Carvalhais2014_FullDepth_full.nc'), decode_times=False)['c_soil'].values[0]
c_soils_1 = np.nanmedian(xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_soil_Carvalhais2014_WWWandNCSCD_full.nc'), decode_times=False)['c_soil'].values.reshape(-1, 360, 720), axis = 0)
c_soils_fd = np.nanmedian(xr.open_dataset(os.path.join(top_dir,'Observation/tau_carvalhais2014_cube/c_soil_Carvalhais2014_FullDepth_full.nc'), decode_times=False)['c_soil'].values.reshape(-1, 360, 720), axis = 0)
c_soils_1 = _apply_a_mask(c_soils_1, all_mask)
c_soils_fd = _apply_a_mask(c_soils_fd, all_mask)

plt.subplot(2,2,4)
plt.scatter((full_depth-1).flatten(), (c_soils_1-c_soils_fd).flatten(),
                    s=0.23,
                    c=arI.flatten(),
                    cmap=cm_rat,
                    norm=mpl.colors.BoundaryNorm(_aridity_bounds,
                                                len(_aridity_bounds)),
                    linewidths=0.3218,
                    alpha=0.274)
plt.axhline(y=0,lw=0.7,color='k')
plt.axvline(x=0,lw=0.7,color='k')
plt.xlabel('soil depth - 1')
plt.ylabel("csoil$_{1m}$ - csoil$_{fd}$")
plt.title('WWWandNCSCD')


plt.savefig('xtra_fig_soil_depth_vs_del1m-fd_cSoil.png',            bbox_inches='tight',dpi=350)
kera
plt.show()




print(f)

plt.figure()
plt.imshow(full_depth)
plt.colorbar()

plt.figure()
plt.imshow(full_depth < 1)
plt.colorbar()
plt.show()


plt.show()



# indat0 = np.array(f.get(_var)) #


# soil_depth = 
#-------------------------------------------------------------------------------## mask for MODELS ONLY
#-------------------------------------------------------------------------------

# all_mod_mask = get_data_mask(all_mod_dat_tau_c, fill_val=fill_val, which_mask = all_mask)

# create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid.nc'), 'mask', all_mod_mask.astype(int))

# plt.figure()
# plt.imshow(all_mod_mask)
# plt.colorbar()

