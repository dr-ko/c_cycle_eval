import sys
import _shared_plot as ptool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
from string import ascii_letters as al
from _shared import _get_set, _get_aux_data, _get_data


#-------------------------------------------
## Zonal mean and range
#-------------------------------------------
def get_zonal_range(_datgpp, _area_dat, zonal_set):
    _lats = zonal_set['lats']
    bandsize_mean = zonal_set['bandsize_mean']
    min_points = zonal_set['min_points']
    _latint = abs(_lats[1] - _lats[0])
    __dat = np.ones((np.shape(_datgpp)[0])) * np.nan
    __dat5 = np.ones((np.shape(_datgpp)[0])) * np.nan
    __dat95 = np.ones((np.shape(_datgpp)[0])) * np.nan
    windowSize = int(np.round(bandsize_mean / (_latint * 2)))
    v_mask = ~np.ma.masked_invalid(_datgpp).mask
    nvalids = np.sum(v_mask, axis=1)
    lat_indices = np.argwhere(nvalids > 0)
    for _latInd in lat_indices:
        li = _latInd[0]
        istart = max(0, li - windowSize)
        iend = min(np.size(_lats), li + windowSize + 1)
        _areaZone = _area_dat[istart:iend, :]
        _gppZone = _datgpp[istart:iend, :] * _areaZone
        _datgppZoneNA = _datgpp[istart:iend, :]
        v_mask = ~np.ma.masked_invalid(_gppZone).mask
        nvalids = np.sum(v_mask)
        if nvalids > min_points:
            __dat[li] = np.nansum(_gppZone) / np.nansum(_areaZone)
            __dat5[li] = np.nanpercentile(_datgppZoneNA, 5)
            __dat95[li] = np.nanpercentile(_datgppZoneNA, 95)
    return __dat, __dat5, __dat95


#-------------------------------------------
# plotting
#-------------------------------------------


def plot_zonal_clim(_sp, plot_var, clim_dat, area_dat, zonal_set):
    dat_obs_zonal, dat_obs_zonal_5, dat_obs_zonal_95 = get_zonal_range(
        clim_dat, area_dat, zonal_set)
    lats = zonal_set['lats']
    _sp.plot(dat_obs_zonal, lats, color='k', lw=lwMainLine, label='Obs-based')
    _sp.fill_betweenx(lats,
                      dat_obs_zonal_5,
                      dat_obs_zonal_95,
                      facecolor='grey',
                      alpha=0.40)

    plt.gca().tick_params(labelsize=ax_fs * 0.91)

    # get the multimodel ensemble tau from multimodel ensemble gpp and c_total
    # set the scale and ranges of axes
    valrange_md = obs_dict[plot_var]['plot_range_zonal']
    if plot_var in ['tau_c', 'c_total']:
        plt.gca().set_xscale('log')
    plt.xlim(valrange_md[0], valrange_md[1])
    plt.ylim(-60, 85)
    plt.axhline(y=0, lw=0.48, color='grey')
    plt.axvline(x=0, lw=0.48, color='grey')
    return


#parameters of the figure

lwMainLine = 1.03

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.8
co_settings['fig_settings']['ax_fs'] = ax_fs
zonal_set = fig_set['zonal']
zonal_set['lats'] = np.linspace(-89.75, 89.75, 360, endpoint=True)[::-1]
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_tasp = _get_data('tasp', co_settings, _co_mask=all_mask)

# define figure
fig = plt.figure(figsize=(4.1, 5))
plt.subplots_adjust(hspace=0.2, wspace=0.3)
plt.gca().tick_params(labelsize=ax_fs * 0.91)
tit_x = 0.25
tit_y = 1.0
#-------------------------------------------
# Loop through the variables
#-------------------------------------------
variables = 'pr tasp'.split()
spInd = 1
for _variable in variables:
    # zonal means
    sp1 = plt.subplot(1, 2, spInd)
    if _variable == 'pr':
        clim_dat = all_pr['lpj']
    else:
        clim_dat = all_tasp['lpj']
    plot_zonal_clim(sp1, _variable, clim_dat, area_dat, zonal_set)
    if spInd == 2:
        ptool.rem_ticklabels(which_ax='y')
    h = plt.title(al[spInd - 1] + ') ' + obs_dict[_variable]['title'] + ' (' +
                  obs_dict[_variable]['unit'] + ')',
                  x=tit_x,
                  y=tit_y,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    if spInd == 1:
        plt.ylabel('Latitude ($^\\circ N$)', fontsize=ax_fs, ma='center')
    ptool.rem_axLine(['top', 'right'])
    spInd = spInd + 1

#-------------------------------------------
# save the figure
#-------------------------------------------
t_x = plt.figtext(0.96, 1.038, ' ', transform=plt.gca().transAxes)
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x],
            dpi=fig_set['fig_dpi'])
plt.close(1)
