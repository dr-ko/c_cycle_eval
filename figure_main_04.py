import sys, os, os.path
sys.path.append(os.path.expanduser('./plot_py3/'))
import numpy as np
from string import ascii_letters as al
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['hatch.color'] = 'yellow'
mpl.rcParams['hatch.linewidth'] = 0.7
from _shared import _get_set, _apply_a_mask, _get_aux_data, _get_colormap_info, _get_data, _get_obs_percentiles, _rem_nan
import _shared_plot as ptool


def _fix_map(axis_obj):
    """
    Beautify map object.

    Clean boundaries, coast lines, and removes the outline box/circle.
    """
    axis_obj.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    axis_obj.coastlines(linewidth=0.4, color='grey')
    plt.gca().outline_patch.set_visible(False)
    axis_obj.patch.set_visible(False)
    axis_obj.background_patch.set_alpha(0)
    return axis_obj


def _get_agreement_mask(mmdat, dat_5, dat_95, nmodel_reject=2):
    """
    Get mask of multimodel agreement.

    Finds regions where fewer than one quarter of the model
    simulations are outside the range of observational uncertainty.
    """
    _maskf = np.zeros_like(mmdat)
    _maskf[(mmdat < dat_95) & (mmdat > dat_5)] = 1
    num_count = _maskf.sum(0)
    agreement_mask = np.zeros_like(num_count)
    agreement_mask[num_count < nmodel_reject] = 1
    wnan = np.ma.masked_invalid(dat_5).mask
    agreement_mask[wnan] = 0.
    return agreement_mask


def plot_variable_maps(x0,
                       plotVar,
                       dat_obs,
                       dat_mod,
                       mm_bias,
                       uncer_mask,
                       mesh_x,
                       mesh_y,
                       _co_settings,
                       put_ratio_colorbar=False,
                       put_title=False,
                       title_index=0):
    fig_set = _co_settings['fig_settings']
    ax_fs = fig_set['ax_fs']

    y0 = 0.06
    ysp = 0.04
    xsp = -0.0
    hp = 0.325
    wp = .57
    x00 = 0.4
    y00 = 0.07
    cbxsp = -0.03
    cbysp = 0.04
    hcolo = 0.017 * hp
    wcolo = 0.2

    _bounds, _colmap, _cbticks, _cblabels = _get_colormap_info(plotVar,
                                                               _co_settings,
                                                               isratio=False)
    _ax = plt.axes([x0, y0 + (3) * (hp + ysp), wp, hp],
                   projection=ccrs.Robinson(central_longitude=0),
                   frameon=False)
    _fix_map(_ax)

    plt.imshow(np.ma.masked_less(dat_obs[0:300, :], -999.),
               interpolation='none',
               norm=mpl.colors.BoundaryNorm(_bounds,
                                            len(_bounds) - 1),
               cmap=_colmap,
               origin='upper',
               transform=ccrs.PlateCarree(),
               extent=[-180, 180, -60, 90])
    h = plt.title(al[title_index],
                  x=0.15,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    if put_title:
        plt.text(-106,
                 -35,
                 "Obs-based",
                 fontsize=0.9 * ax_fs,
                 ha='center',
                 transform=ccrs.PlateCarree())

    cbax = [
        x0 + xsp + 0.25 * wp + 0.1 * cbxsp,
        y0 + (4) * (hp + ysp) + cbysp - 0.035, wp * 0.5577, 2.1 * 0.008151
    ]

    varName = obs_dict[plotVar]['title']
    cb_tit_d = varName
    ptool.mk_colo_tau_c(cbax,
                      _bounds,
                      _colmap,
                      tick_locs=_cbticks,
                      cbfs=ax_fs,
                      cbtitle=cb_tit_d,
                      cb_or='horizontal',
                      cbrt=0)
    plt.gca().yaxis.set_ticks_position('left')

    # multimodel ensemble
    _ax = plt.axes([x0, y0 + (2) * (hp + ysp), wp, hp],
                   projection=ccrs.Robinson(central_longitude=0),
                   frameon=False)
    _fix_map(_ax)

    plt.imshow(np.ma.masked_less(dat_mod[0:300, :], -999.),
               interpolation='none',
               norm=mpl.colors.BoundaryNorm(_bounds,
                                            len(_bounds) - 1),
               cmap=_colmap,
               origin='upper',
               transform=ccrs.PlateCarree(),
               extent=[-180, 180, -60, 90])

    h = plt.title(al[title_index + 1],
                  x=0.15,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    if put_title:
        plt.text(-106,
                 -35,
                 "Multimodel\nEnsemble",
                 fontsize=0.9 * ax_fs,
                 ha='center',
                 transform=ccrs.PlateCarree())

    # multimodel bias

    _ax = plt.axes([x0, y0 + (1) * (hp + ysp), wp, hp],
                   projection=ccrs.Robinson(central_longitude=0),
                   frameon=False)
    _fix_map(_ax)

    _bounds, _colmap, _cbticks, _cblabels = _get_colormap_info(plotVar,
                                                               _co_settings,
                                                               isratio=True)

    plt.imshow(np.ma.masked_less(mm_bias[0:300], 0.),
               norm=mpl.colors.BoundaryNorm(boundaries=_bounds,
                                            ncolors=len(_bounds)),
               origin='upper',
               cmap=_colmap,
               transform=ccrs.PlateCarree(),
               extent=[-180, 180, -60, 90])

    # print("hatching started")

    pc = plt.contourf(mesh_x,
                      mesh_y,
                      uncer_mask[0:300],
                      levels=[0, 0.5, 1],
                      alpha=0.,
                      hatches=['', '//////'],
                      transform=ccrs.PlateCarree())

    # print("hatching ended")

    _fix_map(_ax)

    h = plt.title(al[title_index + 2],
                  x=0.15,
                  weight='bold',
                  fontsize=ax_fs * 1.1,
                  rotation=0)
    if put_title:
        plt.text(-106,
                 -35,
                 "Bias and\nAgreement",
                 fontsize=0.9 * ax_fs,
                 ha='center',
                 transform=ccrs.PlateCarree())
    # ## colorbar for ratio
    if put_ratio_colorbar:
        cbax = [
            x0 + xsp - 0.384285 * wp + 0.1 * cbxsp,
            y0 + (1) * (hp + ysp) + cbysp - 0.085, wp * 0.577, 2.1 * 0.008151
        ]
        varName = obs_dict[plotVar]['title']
        cb_tit = '$\\frac{model}{obs-based}$'

        cb = ptool.mk_colo_cont(cbax,
                                _bounds,
                                _colmap,
                                cbfs=ax_fs,
                                cbrt=0,
                                cbtitle=cb_tit,
                                col_scale='log',
                                cb_or='horizontal',
                                tick_locs=_cbticks)

        cb.ax.set_xticklabels(_cblabels, ha='center', rotation=0)

    return


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']

fig_set = co_settings['fig_settings']
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

#get the data of precip and tair from both models and obs
mask_all = 'model_valid_tau_cObs'.split()

aridity_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
_aridity_bounds = co_settings[co_settings['fig_settings']['eval_region']]['bounds']
color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
cm_rat = mpl.colors.ListedColormap(color_list)

all_mask, arI, area_dat = _get_aux_data(co_settings)

# get the model data of the variable of interest
all_mod_dat_tau_c = _get_data('tau_c', co_settings, _co_mask=all_mask)
all_mod_dat_gpp = _get_data('gpp', co_settings, _co_mask=all_mask)
all_mod_dat_c_total = _get_data('c_total',
                                         co_settings,
                                         _co_mask=all_mask)

fill_val = co_settings['fill_val']
perc_range = co_settings['fig_settings']['zonal']['perc_range']

##################################TAUTAUTAU################

models = list(model_dict.keys())
models.insert(0, 'obs')
nmodels = len(models)

###GET All the data

tau_obs = all_mod_dat_tau_c['obs']
tau_obs_5, tau_obs_95 = _get_obs_percentiles('tau_c',
                                             co_settings,
                                             perc_range,
                                             _co_mask=all_mask)
##################GPPGPPPGPP######################

gpp_obs = all_mod_dat_gpp['obs']
gpp_obs_5, gpp_obs_95 = _get_obs_percentiles('gpp',
                                             co_settings,
                                             perc_range,
                                             _co_mask=all_mask)

################### c_total

c_total_obs = all_mod_dat_c_total['obs']
c_total_obs_5, c_total_obs_95 = _get_obs_percentiles('c_total',
                                                   co_settings,
                                                   perc_range,
                                                   _co_mask=all_mask)

mm_full_tau_c = np.ones(
    (nmodels - 1, np.shape(tau_obs)[0], np.shape(tau_obs)[1])) * fill_val
mm_full_gpp = np.ones(
    (nmodels - 1, np.shape(tau_obs)[0], np.shape(tau_obs)[1])) * fill_val
mm_full_c_total = np.ones(
    (nmodels - 1, np.shape(tau_obs)[0], np.shape(tau_obs)[1])) * fill_val

for row_m in range(1, nmodels):
    row_mod = models[row_m]
    mod_dat_row_tau_c = all_mod_dat_tau_c[row_mod]
    mm_full_tau_c[row_m - 1] = mod_dat_row_tau_c

    mod_dat_row_gpp = all_mod_dat_gpp[row_mod]
    mm_full_gpp[row_m - 1] = mod_dat_row_gpp

    mod_dat_row_c_total = all_mod_dat_c_total[row_mod]
    mm_full_c_total[row_m - 1] = mod_dat_row_c_total

# ---- get the uncertainty mask

uncer_mask_tau_c = _get_agreement_mask(mm_full_tau_c,
                                     tau_obs_5,
                                     tau_obs_95,
                                     nmodel_reject=int(nmodels / 4))
print('nPixInValidTau', np.nansum(uncer_mask_tau_c), np.nansum(all_mask), 100 * np.nansum(uncer_mask_tau_c) / np.nansum(all_mask))
uncer_mask_gpp = _get_agreement_mask(mm_full_gpp,
                                     gpp_obs_5,
                                     gpp_obs_95,
                                     nmodel_reject=int(nmodels / 4))
print('nPixInValidgpp', np.nansum(uncer_mask_gpp), np.nansum(all_mask), 100 * np.nansum(uncer_mask_gpp) / np.nansum(all_mask))

uncer_mask_c_total = _get_agreement_mask(mm_full_c_total,
                                        c_total_obs_5,
                                        c_total_obs_95,
                                        nmodel_reject=int(nmodels / 4))
print('nPixInValidc_total', np.nansum(uncer_mask_c_total), np.nansum(all_mask), 100 * np.nansum(uncer_mask_c_total) / np.nansum(all_mask))

mm_tau_c = _rem_nan(np.nanmedian(mm_full_tau_c, axis=0))
mm_gpp = _rem_nan(np.nanmedian(mm_full_gpp, axis=0))

mm_bias_tau_c = mm_tau_c / tau_obs
mm_bias_tau_c = _rem_nan(mm_bias_tau_c)
mm_bias_tau_c = _apply_a_mask(mm_bias_tau_c, all_mask)

mm_gpp = _rem_nan(np.nanmedian(mm_full_gpp, axis=0))
mm_bias_gpp = mm_gpp / gpp_obs
mm_bias_gpp = _rem_nan(mm_bias_gpp)
mm_bias_gpp = _apply_a_mask(mm_bias_gpp, all_mask)

mm_c_total = _rem_nan(np.nanmedian(mm_full_c_total, axis=0))
mm_bias_c_total = mm_c_total / c_total_obs
mm_bias_c_total = _rem_nan(mm_bias_c_total)
mm_bias_c_total = _apply_a_mask(mm_bias_c_total, all_mask)

lats = np.linspace(-59.75, 89.75, 300, endpoint=True)[::-1]
lons = np.linspace(-179.75, 179.75, 720, endpoint=True)

latint = lats[1] - lats[0]
lonint = lons[1] - lons[0]
x, y = np.meshgrid(lons - lonint / 2, lats - latint / 2)


fig = plt.figure(figsize=(9, 6))
x0 = 0.1
plot_variable_maps(x0,
                   'tau_c',
                   tau_obs,
                   mm_tau_c,
                   mm_bias_tau_c,
                   uncer_mask_tau_c,
                   x,
                   y,
                   co_settings,
                   put_title=True,
                   title_index=0)

x0 = 0.48503
plot_variable_maps(x0,
                   'gpp',
                   gpp_obs,
                   mm_gpp,
                   mm_bias_gpp,
                   uncer_mask_gpp,
                   x,
                   y,
                   co_settings,
                   put_title=False,
                   title_index=3)

x0 = 0.8692503
plot_variable_maps(x0,
                   'c_total',
                   c_total_obs,
                   mm_c_total,
                   mm_bias_c_total,
                   uncer_mask_c_total,
                   x,
                   y,
                   co_settings,
                   put_ratio_colorbar=True,
                   put_title=False,
                   title_index=6)

t_x = plt.figtext(0.5, 0.5, ' ', transform=plt.gca().transAxes)

plt.savefig(co_settings['fig_settings']['fig_dir'] + 'fig_' + fig_num + co_settings['exp_suffix'] + '.' + fig_set['fig_format'],
            bbox_inches='tight',
            bbox_extra_artists=[t_x],
            dpi=fig_set['fig_dpi'])
