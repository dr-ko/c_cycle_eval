import sys
import _shared_plot as ptool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = '#888888'
import json
from string import ascii_letters as al
from _shared import _get_set, _get_aux_data, _get_data, _draw_legend_models, _plot_correlations


def get_var_name(_var, obs_dict):
    var_name = obs_dict[_var]['title']
    if "$" in var_name:
        var_name = var_name.replace("$", "")
    return var_name


# get the settings

co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']
nmodels = len(models)

fig_set = co_settings['fig_settings']
ax_fs = fig_set['ax_fs'] * 0.6
fig_set['ax_fs'] = ax_fs

fig_set['lwMainLine'] = 1.03
fig_set['lwModLine'] = 0.45
fig_set['mod_colors'] = co_settings['model']['colors']
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

co_settings['fig_settings']['ax_fs'] = ax_fs

zonal_set = fig_set['zonal']
zonal_set['lats'] = np.linspace(-89.75, 89.75, 360, endpoint=True)[::-1]

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_pr = _get_data('pr', co_settings, _co_mask=all_mask)
all_tas = _get_data('tas', co_settings, _co_mask=all_mask)
all_gpp = _get_data('gpp', co_settings, _get_full_obs=True)
all_c_total = _get_data('c_total', co_settings, _get_full_obs=True)
all_pr['obs'] = all_pr['lpj']
all_tas['obs'] = all_tas['lpj']

all_data = {}
all_data['pr'] = all_pr
all_data['tas'] = all_tas
all_data['c_total'] = all_c_total
all_data['gpp'] = all_gpp
mod_colors = co_settings['model']['colors']

# define figure
fig = plt.figure(figsize=(5, 2.5))
plt.subplots_adjust(hspace=0.2, wspace=0.3)
plt.tick_params(labelsize=ax_fs * 0.6)
tit_x = 0.25
tit_y = 1.0
#-------------------------------------------
# plot c_total-gpp correlation
#-------------------------------------------

sp = plt.subplot(1, 3, 1)
var_info = {}
var_info['x'] = 'c_total'
var_info['y'] = 'gpp'
var_info['z'] = []
var_name_x = get_var_name(var_info['x'], obs_dict)
var_name_y = get_var_name(var_info['y'], obs_dict)
x_lab = '$r_{' + var_name_x + '-' + var_name_y + '}$'
h = plt.title(al[0] + ') ' + x_lab,
              x=tit_x,
              y=tit_y,
              weight='bold',
              fontsize=ax_fs * 1.1,
              rotation=0)
plt.ylabel('Latitude ($^\\circ N$)', fontsize=ax_fs, ma='center')
plt.gca().tick_params(labelsize=ax_fs * 0.91)
_plot_correlations(sp, all_data, var_info, zonal_set, fig_set, co_settings)

leg = _draw_legend_models(co_settings, loc_a=(0.1744, 1.083))

#--------------------------------------------------
# plot c_total-gpp correlation controlled for precip
#--------------------------------------------------
sp = plt.subplot(1, 3, 2)
var_info = {}
var_info['x'] = 'c_total'
var_info['y'] = 'gpp'
var_info['z'] = ['pr']
var_name_x = get_var_name(var_info['x'], obs_dict)
var_name_y = get_var_name(var_info['y'], obs_dict)
var_name_z = get_var_name(var_info['z'][0], obs_dict)
x_lab = '$r_{' + var_name_x + '-' + var_name_y + ', ' + var_name_z + '}$'
h = plt.title(al[1] + ') ' + x_lab,
              x=tit_x,
              y=tit_y,
              weight='bold',
              fontsize=ax_fs * 1.1,
              rotation=0)
ptool.rem_ticklabels(which_ax='y')
plt.gca().tick_params(labelsize=ax_fs * 0.91)

_plot_correlations(sp, all_data, var_info, zonal_set, fig_set, co_settings)

#-------------------------------------------------------
# plot c_total-gpp correlation controlled for temperature
#-------------------------------------------------------
sp = plt.subplot(1, 3, 3)
var_info = {}
var_info['x'] = 'c_total'
var_info['y'] = 'gpp'
var_info['z'] = ['tas']
var_name_x = get_var_name(var_info['x'], obs_dict)
var_name_y = get_var_name(var_info['y'], obs_dict)
var_name_z = get_var_name(var_info['z'][0], obs_dict)
x_lab = '$r_{' + var_name_x + '-' + var_name_y + ', ' + var_name_z + '}$'
h = plt.title(al[2] + ') ' + x_lab,
              x=tit_x,
              y=tit_y,
              weight='bold',
              fontsize=ax_fs * 1.1,
              rotation=0)
ptool.rem_ticklabels(which_ax='y')
plt.gca().tick_params(labelsize=ax_fs * 0.91)

_plot_correlations(sp, all_data, var_info, zonal_set, fig_set, co_settings)

#-------------------------------------------
# save the figure
#-------------------------------------------
t_x = plt.figtext(0.96, 1.038, ' ', transform=plt.gca().transAxes)
plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x, leg],
            dpi=fig_set['fig_dpi'])
plt.close(1)
