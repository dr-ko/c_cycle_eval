import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import json

from _shared import _get_set, _get_data
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
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=None)

# add observation to models list
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']


masks_dir = os.path.join(top_dir, './masks_n/_data/')
os.makedirs(masks_dir, exist_ok=True)


all_mask = np.ones((360, 720))
fill_val = co_settings['fill_val']
create_nc_hlf(os.path.join(masks_dir, 'mask_none.nc'), 'mask', all_mask.astype(int))

#-------------------------------------------------------------------------------## mask for MODELS ONLY
#-------------------------------------------------------------------------------

all_mod_mask = get_data_mask(all_mod_dat_tau_c, fill_val=fill_val, which_mask = all_mask)

create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid.nc'), 'mask', all_mod_mask.astype(int))

plt.figure()
plt.imshow(all_mod_mask)
plt.colorbar()

# get the tau data with observation
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=None)

#-------------------------------------------------------------------------------## mask for TAU obs and models
#------------------------------------------------------
all_mod_mask = get_data_mask(all_mod_dat_tau_c, fill_val=fill_val, which_mask = all_mod_mask)

create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid_tau_cObs.nc'), 'mask', all_mod_mask.astype(int))

plt.figure()
plt.imshow(all_mod_mask)
plt.colorbar()

#------------------------------------------------------
## mask for TAU obs and models, and KG class not desert
#------------------------------------------------------
all_mask_kg = np.copy(all_mod_mask)
datfile = os.path.join(top_dir, 'Koeppen_11Classes_hlf.nc')
datVar = 'KGC'
kgc_dat_f = xr.open_dataset(datfile, decode_times=False)
datKGC = kgc_dat_f[datVar].values
all_mask_kg[datKGC == 3] = 0.

# plt.show()
create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid_tau_cObs_KGBW.nc'), 'mask', all_mask_kg.astype(int))

plt.figure()
plt.imshow(all_mask_kg)
plt.colorbar()

#------------------------------------------------------
## mask for gpp, and tau obs and models
#------------------------------------------------------
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mod_mask)

gpp_obs = all_mod_dat_gpp['obs']

for gppLim in [1, 10, 100]:
    all_mask_gpp = np.copy(all_mod_mask)
    all_mask_gpp[gpp_obs < gppLim / 1000.] = 0.
    create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid_tau_cObs_gppObsgt' +
                    str(int(gppLim)) + 'GC.nc'), 'mask', all_mask_gpp.astype(int))

#------------------------------------------------------
## mask for PRECIP LIMITS
#------------------------------------------------------

all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mod_mask)

pr_obs = all_mod_dat_pr['obs']

for prLim in [50, 100, 250]:
    all_mask_pr = np.copy(all_mod_mask)
    all_mask_pr[pr_obs < prLim] = 0.
    create_nc_hlf(os.path.join(masks_dir, 'mask_model_valid_tau_cObs_prObsgt' +
                    str(int(prLim)) + 'mm.nc'), 'mask', all_mask_gpp.astype(int))