import sys
import _shared_plot as ptool
import string as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
from _shared import _get_set, _apply_common_mask_g, _compress_invalid, _density_estimation, _fit_odr, _get_aux_data, _get_data

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
model_dict = co_settings['model_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.8
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs
mask_all = 'model_valid_tau_cObs'.split()

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_lista = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_lista)

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_mod_dat_tas = _get_data('tas',
                                          co_settings,
                                          _co_mask=all_mask)
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask)
all_mod_dat_ctot = _get_data('c_total',
                                       co_settings,
                                       _co_mask=all_mask)

nmodels = len(models)
biasMax = 4
mod_arI_0 = arI.flatten()

gpp_obs_0 = all_mod_dat_gpp['obs'].flatten()
c_totalobs_0 = all_mod_dat_ctot['obs'].flatten()

ax_fs = 5.6
cm_rat_a = mpl.colors.ListedColormap(color_lista)
cbticks_rat = [1, 2, 3, 4]
fig = plt.figure(figsize=(5, 5.6))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
for row_m in range(nmodels):
    row_mod = models[row_m]
    print(row_mod)
    if row_mod != 'obs':
        mod_dat_gpp_0 = all_mod_dat_gpp[row_mod].flatten()
        mod_dat_c_total0 = all_mod_dat_ctot[row_mod].flatten()
        mod_dat_gpp, mod_arI, mod_dat_ctot, gpp_obs, c_totalobs = _apply_common_mask_g(
            mod_dat_gpp_0, mod_arI_0, mod_dat_c_total0, gpp_obs_0, c_totalobs_0)

        gpp_bias = mod_dat_gpp / gpp_obs
        c_totalbias = mod_dat_ctot / c_totalobs
        gpp_bias = np.ma.masked_outside(gpp_bias, 0, biasMax).filled(np.nan)
        c_totalbias = np.ma.masked_outside(c_totalbias, 0,
                                          biasMax).filled(np.nan)
        # start plotting
        plt.subplot(3, 3, row_m)
        ptool.ax_orig(axfs=ax_fs * 0.9)
        dat1h, dat2h = _apply_common_mask_g(gpp_bias, c_totalbias)
        dat1h = _compress_invalid(dat1h)
        dat2h = _compress_invalid(dat2h)
        print(dat1h, dat2h)
        # X, Y, Z = _density_estimation(dat1h.compressed(), dat2h.compressed())
        X, Y, Z = _density_estimation(dat1h, dat2h)

        # Add contour lines

        plt.title(model_dict[row_mod]['model_name'] + ' (' + str(
            int(
                len(dat1h) * 100. /
                np.ma.count(np.ma.masked_invalid(mod_dat_gpp_0)))) + '%)',
                  fontsize=ax_fs,
                  y=0.95)
        slope, intercept = _fit_odr(dat1h, dat2h)
        if intercept >= 0:
            eqnTxt = "y=" + str(round(slope, 2)) + 'x + ' + str(
                round(intercept, 2))
        else:
            eqnTxt = "y=" + str(round(slope, 2)) + 'x - ' + str(
                abs(round(intercept, 2)))

        plot_x = np.linspace(
            min(np.nanpercentile(dat1h, 5), np.nanpercentile(dat2h, 5)),
            max(np.nanpercentile(dat1h, 95), np.nanpercentile(dat2h, 95)), 100)
        plot_y = intercept + slope * plot_x
        plt.plot(plot_x,
                 plot_y,
                 'k',
                 lw=0.85,
                 ls=':',
                 label="Global",
                 zorder=5)
        _xDat = gpp_bias
        _yDat = c_totalbias
        im = plt.scatter(_xDat,
                         _yDat,
                         s=0.03,
                         c=mod_arI,
                         cmap=cm_rat_a,
                         norm=mpl.colors.BoundaryNorm(_aridity_bounds,
                                                      len(_aridity_bounds)),
                         linewidths=0.3218,
                         alpha=0.274)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('starting ', row_mod)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        for _tr in range(len(_aridity_bounds) - 1):
            tas1 = _aridity_bounds[_tr]
            tas2 = _aridity_bounds[_tr + 1]
            mod_tas_tmp = np.ma.masked_inside(mod_arI, tas1, tas2).mask
            perVal = 2
            _yDat_tr = _yDat[mod_tas_tmp]
            _xDat_tr = _xDat[mod_tas_tmp]
            med_x = _xDat_tr
            med_y = _yDat_tr
            fit_x, fit_y = _apply_common_mask_g(med_x, med_y)
            fit_x = _compress_invalid(fit_x)
            fit_y = _compress_invalid(fit_y)
            slope, intercept = _fit_odr(fit_x, fit_y)
            print(aridity_list[_tr], slope, intercept)
            print('----------------------------')
            plot_x = np.linspace(
                min(np.nanpercentile(fit_x, 5), np.nanpercentile(fit_y, 5)),
                max(np.nanpercentile(fit_x, 95), np.nanpercentile(fit_y, 95)),
                100)
            plot_y = intercept + slope * plot_x
            plt.plot(plot_x,
                     plot_y,
                     c=color_lista[_tr],
                     ls='-',
                     lw=0.8,
                     marker=None,
                     label=aridity_list[_tr])
        plt.xlim(-0.01, biasMax)
        plt.ylim(-0.01, biasMax)
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()
        plt.plot((xmin, xmax), (ymin, ymax),
                 '#555555',
                 lw=0.5,
                 label='$\\frac{\\tau\ (mod)}{\\tau\ (obs)}$=1',
                 zorder=-2)
        plt.axhline(y=1,
                    color='#555555',
                    ls='-.',
                    lw=0.5,
                    label='$\\frac{C_{total}\ (mod)}{C_{total}\ (obs)}$=1',
                    zorder=-2)
        plt.axvline(x=1,
                    color='#555555',
                    ls='--',
                    lw=0.5,
                    label='$\\frac{GPP\ (mod)}{GPP\ (obs)}$=1',
                    zorder=-2)
        plt.contour(X, Y, Z, colors='white', linewidths=0.5, corner_mask=False)

        plt.plot(1, 1, 'k', marker='x', lw=0, markersize=3, mew=0.9)
        if row_m == 7:
            plt.legend(loc=(1.1, 0.05),
                       ncol=1,
                       fontsize=1.03 * ax_fs,
                       frameon=False)
            plt.xlabel('$\\frac{GPP\ (mod)}{GPP\ (obs)}$', fontsize=1.03 * ax_fs)
            plt.ylabel('$\\frac{C_{total}\ (mod)}{C_{total}\ (obs)}$',
                       fontsize=1.03 * ax_fs)
        t_t = plt.text(0.025,
                       1.028,
                       st.ascii_letters[row_m - 1],
                       weight='bold',
                       fontsize=ax_fs,
                       transform=plt.gca().transAxes)
        ptool.put_ticks(nticks=4, which_ax = 'both')

t_x = plt.figtext(0.5, 0.5, ' ', transform=plt.gca().transAxes)

plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x],
            dpi=fig_set['fig_dpi'])
plt.close()
