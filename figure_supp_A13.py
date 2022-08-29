import sys
import _shared_plot as ptool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
from string import ascii_letters as al

from _shared import _get_set, _apply_common_mask_g, _get_aux_data, _get_data, _draw_legend_aridity, _fit_least_square
#--------

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.7
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]
co_settings['fig_settings']['ax_fs'] = ax_fs
nmodels = len(models)

#get the data of precip and tair from both models and obs

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas',
                                          co_settings,
                                          _co_mask=all_mask)
all_mod_dat_evapotrans = _get_data('evapotrans', co_settings, _co_mask=all_mask)
all_mod_dat_ra = _get_data('ra', co_settings, _co_mask=all_mask)
all_mod_dat_rh = _get_data('rh', co_settings, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask)
all_mod_dat_mrso = _get_data('mrso', co_settings, _co_mask=all_mask)
all_mod_dat_mrro = _get_data('mrro', co_settings, _co_mask=all_mask)



aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)


nmodels = len(models)
fit_dict = {}

relS = 'tas-mrso,tas-ra,tas-rh,tas-gpp,mrso-ra,mrso-rh,mrso-gpp'.split(',')
nRel = len(relS)
y_a=[1,1+nRel,1+2*nRel,1+3*nRel,1+4*nRel,1+5*nRel]

loc_vars = 'ra rh  mrso mrro'.split()
fit_method = co_settings['fig_settings']['fit_method']
    
### get the relationships for all models
pcoffs_ar = []
for row_m in range(nmodels):
    row_mod = models[row_m]
    mod_gpp = all_mod_dat_gpp[row_mod]
    mod_ra = all_mod_dat_ra[row_mod]
    mod_rh = all_mod_dat_rh[row_mod]
    mod_tas = all_mod_dat_tas['lpj'] #- 273.15
    mod_mrso = all_mod_dat_mrso[row_mod]
    mod_mrro = all_mod_dat_mrro[row_mod]
    mod_evapotrans = all_mod_dat_evapotrans[row_mod]
    mod_mrso, mod_gpp, mod_arI, mod_tas, mod_ra, mod_rh, mod_evapotrans, mod_mrro = _apply_common_mask_g(
        mod_mrso, mod_gpp, arI, mod_tas, mod_ra, mod_rh, mod_evapotrans, mod_mrro)
    for reStr in relS:
        _intercept=True
        varX = reStr.split('-')[0]
        varY = reStr.split('-')[1]
        if row_mod == 'obs':
            pcoffs = reStr +'|obs|'
        else:
            pcoffs = reStr +'|'+model_dict[row_mod]['model_name']+'|'
        if varY == 'gpp':
            _logY = False
        else:
            _logY = False
        if varX == 'pr' and varY == 'gpp':
            _intercept=False
            bnds = [(-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)]
        else:
            bnds = [(0, -np.inf, -np.inf), (np.inf, np.inf, np.inf)]
        if reStr not in list(fit_dict.keys()):
            fit_dict[reStr] = {}
        _xDat = vars()['mod_' + varX]
        _yDat = vars()['mod_' + varY]
        for _tr in range(len(_aridity_bounds) - 1):
            tas1 = _aridity_bounds[_tr]
            tas2 = _aridity_bounds[_tr + 1]
            mod_tas_tmp = np.ma.masked_inside(mod_arI, tas1, tas2).mask
            _yDat_tr = _yDat[mod_tas_tmp]
            _xDat_tr = _xDat[mod_tas_tmp]
            _yDat_tr=_yDat_tr/_yDat_tr.max()
            _xDat_tr=_xDat_tr/_xDat_tr.max()
            _yDat_tr=_yDat_tr/np.nanpercentile(_yDat_tr,98)#.max()
            _xDat_tr=_xDat_tr/np.nanpercentile(_xDat_tr,98)#.max()
            ariName = aridity_list[_tr]
            if ariName not in list(fit_dict[reStr].keys()):
                fit_dict[reStr][ariName] = {}
            if row_mod not in list(fit_dict[reStr][ariName].keys()):
                fit_dict[reStr][ariName][row_mod] = {}

            fit_dat = _fit_least_square(_xDat_tr,
                                        _yDat_tr,
                                        _logY=_logY,
                                        method=fit_method,
                                        _intercept=_intercept,
                                        _bounds=bnds)

            # create the string to write in summary text file
            coffs = fit_dat['coef']
            r2 = fit_dat['metr']['r2']
            r_mad = fit_dat['metr']['r_mad']

            pcoff = "%.2e" % coffs[0] + '|' + "%.2e" % coffs[1] + '|' + str(
                np.round(coffs[2],
                            2)) + '|' '%.2f' % r2 + '|' '%.2f' % r_mad + ''

            xx = fit_dat['pred']['x']
            yy = fit_dat['pred']['y']

            # save the fitted data in dictionary to use for plotting later
            fit_dict[reStr][ariName][row_mod]['coffs'] = coffs
            fit_dict[reStr][ariName][row_mod]['xx'] = xx
            fit_dict[reStr][ariName][row_mod]['yy'] = yy
            pcoffs=pcoffs+pcoff+'|'
        pcoffs_ar=np.append(pcoffs_ar,pcoffs)    
        print(pcoffs_ar.shape)
pcoffs_ar=np.array(pcoffs_ar).reshape(-1,len(relS))
pcoffs_ar = pcoffs_ar.flatten(order='F')
print(pcoffs_ar.shape)
fig = plt.figure(figsize=(5.1, 6.1))
plt.subplots_adjust(hspace=0.4, wspace=0.3)


# loop through models
for row_m in range(nmodels):
    # loop through relationships
    for reStr in relS:
        reInd = relS.index(reStr)
        spInd = len(relS) * (row_m) + reInd + 1
        plt.subplot(nmodels, len(relS), spInd)
        if spInd in y_a:
            ptool.rem_ticklabels(which_ax = 'x')
        elif spInd == 6*nRel + 1:
            ptool.ax_orig(axfs=ax_fs * 0.5)
            ptool.rotate_labels(which_ax = 'x',rot=90,axfs=ax_fs * 0.5)
        elif spInd > 6*nRel + 1:
            ptool.rem_ticklabels(which_ax = 'y')
            ptool.rotate_labels(which_ax = 'x',rot=90,axfs=ax_fs * 0.5)
        else:
            ptool.rem_ticklabels()
        varX = reStr.split('-')[0]
        varY = reStr.split('-')[1]

        if varX == 'tas':
            x_tit = obs_dict['tas']['title']
        else:
            x_tit = obs_dict['mrso']['title']
        y_tit = obs_dict[varY]['title']

        for ariName in aridity_list:
            # arInd = aridity_list.index(ariName)
            row_mod = models[row_m]
            datX = fit_dict[reStr][ariName][row_mod]['xx']
            datY = fit_dict[reStr][ariName][row_mod]['yy']
            coffs = fit_dict[reStr][ariName][row_mod]['coffs']
            arInd = aridity_list.index(ariName)
            print(row_mod,reStr, ariName, spInd)
            # if spInd
            ptool.ax_orig(axfs=ax_fs * 0.9)
            # ptool.rem_ticklabels()
            plt.ylim(0, 1)
            if varY == 'gpp':
                _logY = True
            else:
                _logY = True
            if row_mod == 'obs':
                _lw = 0.55
                mName = '$ra$: C2014,GPP: MTE'
            else:
                _lw = 0.55
                mName = model_dict[row_mod]['model_name']
            plt.plot(datX,
                    datY,
                    ls='-',
                    lw=_lw,
                    color=color_list[arInd],
                    marker=None,
                    label=ariName,
                    zorder=0)
            if varY == 'ra':
                xPos = 0.627
            else:
                xPos = 0.13951

            if row_m == 0 and reInd == 0:
                leg = _draw_legend_aridity(co_settings, loc_a=(1.98227722, 1.37125855), ax_fs = 7)
            if row_m == 0:
                plt.title(y_tit, y=0.98, fontsize=ax_fs)
            if reInd == nRel -1:
                h = plt.ylabel(mName,
                            # weight='bold',
                            color=co_settings['model']['colors'][row_mod],
                            fontsize=ax_fs*0.928,
                            rotation=90)
                plt.gca().yaxis.set_label_position("right")
            if row_m == nmodels-1:
                plt.xlabel(x_tit, fontsize=ax_fs * 0.9)
        ptool.rotate_labels(which_ax = 'x',rot=90,axfs=ax_fs * 0.9)

plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix']+'.'+fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[leg],
            dpi=fig_set['fig_dpi'])
