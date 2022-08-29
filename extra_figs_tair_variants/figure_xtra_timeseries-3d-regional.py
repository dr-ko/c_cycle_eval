import os, sys
import _shared_plot as ptool
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import xarray as xr
#--------
import numpy as np
import matplotlib.pyplot as plt
from string import ascii_letters as al
import pickle

from _shared import _get_set, _apply_a_mask, _get_aux_data, _get_data, _draw_legend_models

#--------
def get_regn_time_series(dat_var,global_mod_dat, co_settings, regn_dat):
    all_mod_dat = {}
    _models = models
    model_var = obs_dict[dat_var]['model_var']
    for _md in _models:
        all_mod_dat[_md]={}
        dat_global = global_mod_dat[_md]
        print(_md)
        for regnn in regns.keys():
            if regnn == 0:
                cond = regn_dat[co_settings['auxiliary']['aridity_var']] > 0
            else:
                cond = regn_dat[co_settings['auxiliary']['aridity_var']] == regnn
            if dat_var in pgc_vars:
                print(list(dat_global.keys()))
                dat_row = dat_global[model_var].where(cond).sum(
                        dim=('lat', 'lon')) * 1.e-12
            else:
                dat_row = dat_global[model_var].where(cond).mean(
                        dim=('lat', 'lon'))
            print(dat_row)
            dat_sel_reg = dat_row.values
            dat_sel_reg = running_mean(dat_sel_reg, 10)
            dat_sel_reg = dat_sel_reg.reshape(-1, s_int).mean(1)
            all_mod_dat[_md][regnn]=dat_sel_reg

            s_years_reg = s_years.reshape(-1, s_int).mean(1)
            plt.plot(s_years_reg, dat_sel_reg, label=regns[regnn])
            plt.title(regns[regnn])
        plt.legend()
        plt.savefig(ts_dir + dat_var+'_'+_md+'.png', bbox_inches='tight', bbox_extra_artists=[], dpi=300)
        plt.close()
    return (all_mod_dat)

def calc_regn_tau_c(mod_c_total, mod_gpp):
    all_mod_dat = {}
    _models = models
    datVar = 'tau_c'
    for _md in _models:
        _mdI = _models.index(_md)
        all_mod_dat[_md]={}
        for regnn in regns.keys():
            all_mod_dat[_md][regnn]=mod_c_total[_md][regnn]/mod_gpp[_md][regnn]
            s_years_reg = s_years.reshape(-1, s_int).mean(1)
            plt.plot(s_years_reg, all_mod_dat[_md][regnn], label=regns[regnn])
            plt.title(regns[regnn])
        plt.legend()
        plt.savefig(ts_dir + datVar+'_'+_md+'.png', bbox_inches='tight', bbox_extra_artists=[], dpi=300)
        plt.close()

        # plt.show()

        # print(modInfo,np.shape(mod_dat))
        # mod_dat = mod_dat * datCorr
        # mod_dat=apply_mask(mod_dat,all_mask)
    return (all_mod_dat)

def running_mean(x, N):
    intV = int(N / 2)
    lenX = len(x)
    outA = np.ones(np.shape(x))
    for _ind in range(lenX):
        _indMin = max(_ind - intV, 0)
        _indMax = min(_ind + intV + 1, lenX)
        # print('llalalala',_ind,_indMin,_indMax)
        outA[_ind] = np.nanmean(x[_indMin:_indMax])
    return (outA)

regns = {
        0: "Global",
        1: "Arid",
        2: "Semi-arid",
        3: "Sub-humid",
        4: "Humid"
    }


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
# models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)
ax_fs = fig_set['ax_fs']


fill_val = co_settings['fill_val']
perc_range = co_settings['fig_settings']['zonal']['perc_range']
pgc_vars = co_settings['pgc_vars']
mod_colors = co_settings['model']['colors']
model_dict = co_settings['model_dict']


s_s_year = co_settings['syear_ts']
e_s_year = co_settings['eyear_ts']
s_years = np.arange(int(s_s_year), int(e_s_year)+1, 1)
s_int  = 1
calc_reg = True
# calc_reg = False
h_space = 0.4
w_space = 0.4
regOrder = [1,2,3,4,0]
ts_dir = os.path.join(co_settings['fig_settings']['fig_dir'], 'timeseries/')
os.makedirs(ts_dir, exist_ok=True)
if calc_reg:
    all_mask, arI, area_dat = _get_aux_data(co_settings)

    # get the model data of the variable of interest
    all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask, _get_model_time_series = True)
    all_mod_dat_tas = _get_data('tas',co_settings,_co_mask=all_mask, _get_model_time_series = True)
    all_mod_dat_c_total = _get_data('c_total', co_settings, _co_mask=all_mask, _get_model_time_series = True)
    all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask, _get_model_time_series = True)
    ari_var = co_settings['auxiliary']['aridity_var']
    arI = xr.open_dataset(
    os.path.join(top_dir, co_settings['auxiliary']['aridity_file']))

    ariVal = arI[ari_var].values
    ari_mask = _apply_a_mask(ariVal, all_mask)
    arI[ari_var].values = ari_mask
    print(list(all_mod_dat_c_total.keys()))
    print(list(all_mod_dat_gpp.keys()))

    all_c_total = get_regn_time_series('c_total', all_mod_dat_c_total, co_settings, arI)
    with open(ts_dir + 'all_mod_dat_c_total.pkl', 'wb') as f:
        pickle.dump(all_c_total, f)

    all_pr = get_regn_time_series('pr', all_mod_dat_pr, co_settings, arI)
    with open(ts_dir + 'all_mod_dat_pr.pkl', 'wb') as f:
        pickle.dump(all_pr, f)

    all_tas = get_regn_time_series('tas', all_mod_dat_tas, co_settings, arI)
    with open(ts_dir + 'all_mod_dat_tas.pkl', 'wb') as f:
        pickle.dump(all_tas, f)

    all_gpp = get_regn_time_series('gpp', all_mod_dat_gpp, co_settings, arI)
    with open(ts_dir + 'all_mod_dat_gpp.pkl', 'wb') as f:
        pickle.dump(all_gpp, f)

    # np.save(ts_dir + 'all_mod_dat_tas.npy',all_mod_dat_tas)
    all_tau_c = calc_regn_tau_c(all_c_total, all_gpp)
    with open(ts_dir + 'all_mod_dat_tau_c.pkl', 'wb') as f:
        pickle.dump(all_tau_c, f)

else:
    with open(ts_dir + 'all_mod_dat_pr.pkl', 'rb') as f:
        all_pr = pickle.load(f)
    with open(ts_dir + 'all_mod_dat_tas.pkl', 'rb') as f:
        all_tas = pickle.load(f)
    with open(ts_dir + 'all_mod_dat_gpp.pkl', 'rb') as f:
        all_gpp = pickle.load(f)
    with open(ts_dir + 'all_mod_dat_c_total.pkl', 'rb') as f:
        all_c_total = pickle.load(f)
    with open(ts_dir + 'all_mod_dat_tau_c.pkl', 'rb') as f:
        all_tau_c = pickle.load(f)

# del ts vs del pr vs tau driven stocks
plt.figure(figsize=(6,9))
plt.subplots_adjust(wspace=0.15,hspace=0.2)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI,projection='3d')
    plt.gca().view_init(elev=25, azim=-45)
    # plt.gca().set_facecolor('k')
    plt.gca().xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().xaxis._axinfo["grid"]['linewidth'] =  0.4
    plt.gca().yaxis._axinfo["grid"]['linewidth'] =  0.4
    plt.gca().zaxis._axinfo["grid"]['linewidth'] =  0.4

    for _md in models:
        # ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(tas-tas[0], pr-pr[0], gpp[0]*(tau-tau[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.2)

        plt.plot(tas-tas[0], pr-pr[0], np.ones_like(pr) * -0.2,'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)

        plt.plot(tas-tas[0], np.ones_like(pr) * 50 , gpp[0]*(tau-tau[0])/c_total[0],'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)

        plt.plot(np.ones_like(pr) * -0.250, pr-pr[0] , gpp[0]*(tau-tau[0])/c_total[0],'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)

        plt.gca().set_zlim(-0.2, 0.05)
        plt.gca().set_xlim(-0.25, 1.25)
        plt.gca().set_ylim(-25, 50)
        plt.title(al[spnI-1]+') '+regns[regnn], x=0.53, y = 0.96, fontsize=ax_fs)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.gca().set_zlabel('$\\frac{\\Delta_{C_{total}}(\\tau)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (-0.8, 50, 0.083), is_3d=True, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_del-pr_tau_cdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del ts vs del pr vs tau driven stocks
plt.figure(figsize=(6,9))
plt.subplots_adjust(wspace=0.15,hspace=0.2)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI,projection='3d')
    plt.gca().view_init(elev=25, azim=-45)
    # plt.gca().set_facecolor('k')
    plt.gca().xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.gca().xaxis._axinfo["grid"]['linewidth'] =  0.4
    plt.gca().yaxis._axinfo["grid"]['linewidth'] =  0.4
    plt.gca().zaxis._axinfo["grid"]['linewidth'] =  0.4

    for _md in models:
        # ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.gca().set_zlim(-0.05, 0.2)
        plt.gca().set_xlim(-0.25, 1.25)
        plt.gca().set_ylim(-25, 50)
        plt.plot(tas-tas[0], pr-pr[0], tau[0] *(gpp-gpp[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.2)

        plt.plot(tas-tas[0], pr-pr[0], np.ones_like(pr) * -0.05,'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)

        plt.plot(tas-tas[0], np.ones_like(pr) * 50 , tau[0] *(gpp-gpp[0])/c_total[0],'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)

        plt.plot(np.ones_like(pr) * -0.250, pr-pr[0] , tau[0] *(gpp-gpp[0])/c_total[0],'--', color=mod_colors[_md], lw = 0, marker = 'o', markerfacecolor='None', markersize=1.2, markeredgewidth=0.1)
        plt.title(al[spnI-1]+') '+regns[regnn], x=0.53, y = 0.96, fontsize=ax_fs)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.gca().set_zlabel('$\\frac{\\Delta_{C_{total}}(GPP)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (-0.8, 50, 0.2583), is_3d=True, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_del-pr_gppdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del ts vs del GPP
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(tas-tas[0], (gpp-gpp[0])/gpp[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{GPP}}{GPP_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.05, 0.3)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_delgpp.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del pr vs del GPP
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(pr-pr[0], (gpp-gpp[0])/gpp[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{GPP}}{GPP_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-25, 50)
        plt.ylim(-0.05, 0.3)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-pr_delgpp.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del ts vs del tau
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(tas-tas[0], (tau-tau[0])/tau[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{\\tau}}{\\tau_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.2, 0.05)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_deltau.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del pr vs del tau
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        # plt.plot(tas-tas[0], gpp[0]*(tau-tau[0]),'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.plot(pr-pr[0], (tau-tau[0])/tau[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{\\tau}}{\\tau_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-25, 50)
        plt.ylim(-0.2, 0.05)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-pr_deltau.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# plt.show()

# del ts vs tau driven stocks
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(tas-tas[0], gpp[0]*(tau-tau[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{C_{total}}(\\tau)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.2, 0.05)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_tau_cdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()
# del ts vs gpp driven stocks
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(tas-tas[0], tau[0]*(gpp-gpp[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{T}\ (^{\\circ} C)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{C_{total}}(GPP)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-0.25, 1.25)
        plt.ylim(-0.05, 0.3)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-tas_gppdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()
# del precip vs tau driven stocks
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(pr-pr[0], gpp[0]*(tau-tau[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{C_{total}}(\\tau)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-25, 50)
        plt.ylim(-0.2, 0.05)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-pr_tau_cdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()

# del precip vs gpp driven stocks
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace=w_space,hspace=h_space)
spnI = 1
for regnn in regOrder:
    plt.subplot(3,2,spnI)
    for _md in models:
        ptool.ax_orig(axfs=ax_fs * 0.9)
        mName = model_dict[_md]['model_name']
        pr = all_pr['lpj'][regnn]
        tas = all_tas['lpj'][regnn]
        tau = all_tau_c[_md][regnn]
        gpp = all_gpp[_md][regnn]
        c_total = all_c_total[_md][regnn]
        plt.plot(pr-pr[0], tau[0]*(gpp-gpp[0])/c_total[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        # plt.plot(pr-pr[0],tau-tau[0],'--', color=mod_colors[_md], label=mName, lw = 0, marker = 'o', markersize=1.19)
        plt.xlabel('$\\Delta_{P}\ (mm/yr)$', ha='center', fontsize=ax_fs * 0.9)
        plt.ylabel('$\\frac{\\Delta_{C_{total}}(GPP)}{C_{0}}$', ha='center', fontsize=ax_fs * 0.9, rotation=90)
        plt.xlim(-25, 50)
        plt.ylim(-0.05, 0.3)
        plt.title(al[spnI-1]+') '+regns[regnn], fontsize=ax_fs)
    if spnI == 1:
        leg = _draw_legend_models(co_settings, loc_a = (.013722, 1.1407125855), is_3d=False, inc_mme=False, inc_obs=False)
    spnI = spnI + 1
plt.savefig(ts_dir + 'del-pr_gppdrivenstock.png', bbox_inches='tight', bbox_extra_artists=[leg], dpi=300)
plt.close()
plt.show()
