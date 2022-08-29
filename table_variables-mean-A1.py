import os, sys
import numpy as np
from _shared import _get_set, _get_aux_data, _get_data

from _shared_regional_funcs import _get_regional_clim, _get_regional_tau_c_range, _get_regional_means, _get_regional_range

biomes_info = {
    '1': 'Arid',
    '2': 'Semi-\narid',
    '3': 'Sub-\nhumid',
    '4': 'Humid'
}

co_settings = _get_set()
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']

# get the auxiliary data
all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas', co_settings, _co_mask=all_mask)
tab_num = sys.argv[0].split('.py')[0].split('_')[-1]


# ------------------------------
# get for climate
# ------------------------------

# clim variables (mean and std)
dat_obs_pr = all_mod_dat_pr['lpj']
dat_obs_tas = all_mod_dat_tas['lpj']
dat_obs_regn_pr = _get_regional_clim(dat_obs_pr, area_dat, arI,
                                        co_settings)
dat_obs_regn_tas = _get_regional_clim(dat_obs_tas, area_dat, arI,
                                        co_settings)


head = '|'+'Global|'+'Arid|'+'Semi-arid|'+'Sub-humid|'+'Humid|'
varibs = 'pr tas'.split()
with open(os.path.join(co_settings['fig_settings']['fig_dir'], 'summary_clim_' + tab_num + co_settings['exp_suffix'] + '.txt'),'w') as f_:
    f_.write(head+'\n')
    for _vrb in varibs:
        dv_mean= vars()['dat_obs_regn_'+_vrb][0,:]
        dv_low= vars()['dat_obs_regn_'+_vrb][1,:]
        dv_hi= vars()['dat_obs_regn_'+_vrb][2,:]
        arStr = _vrb+'|'
        for ar in range(5):
            arStr = arStr + str(np.round(dv_mean[ar],1))+' ('+str(np.round(dv_low[ar],1))+'-'+str(np.round(dv_hi[ar],1))+')' +'|'
            print(ar,arStr)
        f_.write(arStr[:]+'\n')



all_mod_dat_tau_c = _get_data('tau_c', co_settings, _get_full_obs=True)
all_mod_dat_gpp = _get_data('gpp', co_settings, _get_full_obs=True)
all_mod_dat_c_total = _get_data('c_total', co_settings, _get_full_obs=True)
all_mod_dat_c_soil = _get_data('c_soil', co_settings, _get_full_obs=True)
all_mod_dat_c_veg = _get_data('c_veg', co_settings, _get_full_obs=True)

# ------------------------------
# get data for GPP and c_total
# ------------------------------
#%%%%GPP
# dat_obs_gpp = np.nanmedian(all_mod_dat_gpp['obs'], axis=0)
# dat_obs_gpp_full = all_mod_dat_gpp['obs']
# dat_obs_regn_gpp = _get_regional_means('gpp', dat_obs_gpp, area_dat, arI,
#                                         co_settings)
# dat_obs_regn_range_gpp = _get_regional_range_perFirst('gpp', dat_obs_gpp_full,
#                                                        area_dat, arI,
#                                                        co_settings)
dat_obs_gpp_full = all_mod_dat_gpp['obs']
dat_obs_regn_range_gpp, dat_obs_regn_gpp = _get_regional_range(
    'gpp', dat_obs_gpp_full, area_dat, arI, co_settings)

#%%%% c_total
# dat_obs_c_total = np.nanmedian(all_mod_dat_c_total['obs'], axis=0)
# dat_obs_c_total_full = all_mod_dat_c_total['obs']
# dat_obs_regn_c_total = _get_regional_means('c_total', dat_obs_c_total, area_dat,
#                                            arI, co_settings)

dat_obs_c_total_full = all_mod_dat_c_total['obs']
dat_obs_regn_range_c_total, dat_obs_regn_c_total = _get_regional_range(
    'c_total', dat_obs_c_total_full, area_dat, arI, co_settings)

dat_obs_c_soil_full = all_mod_dat_c_soil['obs']
dat_obs_regn_range_c_soil, dat_obs_regn_c_soil = _get_regional_range(
    'c_soil', dat_obs_c_soil_full, area_dat, arI, co_settings)

dat_obs_c_veg_full = all_mod_dat_c_veg['obs']
dat_obs_regn_range_c_veg, dat_obs_regn_c_veg = _get_regional_range(
    'c_veg', dat_obs_c_veg_full, area_dat, arI, co_settings)


# dat_obs_c_total = np.nanmedian(all_mod_dat_c_total['obs'], axis=0)
# dat_obs_c_total_full = all_mod_dat_c_total['obs']
# dat_obs_regn_c_total = _get_regional_means('c_total', dat_obs_c_total, area_dat,
#                                            arI, co_settings)
# dat_obs_regn_range_c_total = _get_regional_range_perFirst(
#     'c_total', dat_obs_c_total_full, area_dat, arI, co_settings)


dat_obs_regn_range_tau_c, dat_obs_regn_tau_c = _get_regional_tau_c_range(
    dat_obs_gpp_full, dat_obs_c_total_full, area_dat, arI, co_settings)

# carbon variables (mean and ranges across ensembles)
varibs = 'gpp c_total tau_c c_soil c_veg'.split()

head = '|'+'Global|'+'Arid|'+'Semi-arid|'+'Sub-humid|'+'Humid|'
with open(os.path.join(co_settings['fig_settings']['fig_dir'], 'summary_carbon_' + tab_num + co_settings['exp_suffix'] + '.txt'),'w') as f_:
    f_.write(head+'\n')
    for _vrb in varibs:
        dv= vars()['dat_obs_regn_'+_vrb]    
        dvr= vars()['dat_obs_regn_range_'+_vrb]
        arStr = _vrb+'|'
        for ar in range(5):
            arStr = arStr + str(np.round(dv[ar],1))+' ('+str(np.round(dvr[0][ar],1))+'-'+str(np.round(dvr[1][ar],1))+')' +'|'              
            print(ar,arStr)
        f_.write(arStr[:]+'\n')
