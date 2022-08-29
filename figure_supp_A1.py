import sys
import cartopy.crs as ccrs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['hatch.color'] = '#888888'
plt.rcParams.update({'font.size': 8})

from _shared import _get_set, _get_aux_data


def mk_glo_map(datBas, _bounds_tmp):
    npoint = np.ma.count(np.ma.masked_invalid(datBas))

    cm2 = mpl.colors.ListedColormap(color_list)
    bounds2 = np.arange(-0.5, len(_bounds_tmp),
                        1)  #([-0.5,0.5, 1.5,2.5,3.5,4.5])
    vegcls = _tickLabs
    vgMap = datBas
    #---------------------------
    plt.figure()
    _ax = plt.subplot(1,
                      1,
                      1,
                      projection=ccrs.Robinson(central_longitude=0),
                      frameon=False)
    plt.imshow(np.ma.masked_less(vgMap[:300, :], -999.),
               cmap=cm2,
               norm=mpl.colors.BoundaryNorm(bounds2, cm2.N),
               interpolation='none',
               origin='upper',
               transform=ccrs.PlateCarree(),
               extent=[-180, 180, -60, 90])
    _ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    _ax.coastlines(linewidth=0.4, color='grey')
    plt.gca().outline_patch.set_visible(False)
    cblab = []
    for vg in range(len(vegcls)):
        lpvgt = np.ma.nonzero(np.ma.masked_not_equal(vgMap, vg + 1).filled(0))
        pcon = str(np.round(len(lpvgt[0]) * 1. / npoint * 100., 1)) + "%"
        cbtxt = vegcls[vg] + '\n(' + pcon + ')'
        cblab = np.append(cblab, cbtxt)
    cb = plt.colorbar(boundaries=bounds2[1:],
                      shrink=.75,
                      aspect=50,
                      drawedges=True,
                      orientation='horizontal',
                      pad=0.01)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(7.52)
        t.set_rotation(0)
    #    cb.set_ticks(arange(colomin,colomax+coloint,coloint))
    ylocss = []
    for jp in cb.ax.xaxis.get_ticklocs():
        print(jp)
        ylocss = np.append(ylocss, jp)
    ylocar = []
    for _yy in range(len(ylocss) - 1):
        ylocar = np.append(ylocar, 0.5 * (ylocss[_yy] + ylocss[_yy + 1]))
    cb.ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ylocar))
    #    cb.ax.yaxis.set_tickloc(FixedLocator(arange(1,6)))
    cb.ax.set_xticklabels(cblab)

    fix_cb(cb, ax_fs=ax_fs)
    t1 = plt.figtext(-0.04, 0.30, ' ', transform=plt.gca().transAxes)
    t2 = plt.figtext(1.03, 1.06, ' ', transform=plt.gca().transAxes)
    plt.savefig(os.path.join(co_settings['fig_settings']['fig_dir'] 
        , svname + '.' + co_settings['fig_settings']['fig_format']),
                bbox_inches='tight',
                bbox_extra_artists=[t1, t2],
                dpi=co_settings['fig_settings']['fig_dpi'])
    plt.close(1)


def fix_cb(_cb, ax_fs=6):
    _cb.ax.tick_params(labelsize=ax_fs, size=2, width=0.3)
    print(_cb.ax.yaxis.get_ticklocs())
    ##hack the lines of the colorbar to make them white, the same color of background so that the colorbar looks broken.
    _cb.outline.set_alpha(0.)
    _cb.outline.set_color('white')
    _cb.outline.set_linewidth(1)
    _cb.dividers.set_linewidth(2)
    _cb.dividers.set_alpha(1.0)
    _cb.dividers.set_color('white')
    for ll in _cb.ax.xaxis.get_ticklines():
        ll.set_alpha(0.)


def fix_cblabel(cb):
    cblabs = cb.ax.yaxis.get_ticklabels()
    xlabs = []
    for __b in cblabs:
        _b = __b.get_text()
        # print(float(_b))
        if _b == '0':
            xlabs = np.append(xlabs, _b)
        else:
            xlabs = np.append(xlabs,
                              '%d' % (round(np.power(10, float(_b)), 0)))

    cb.ax.set_yticklabels(xlabs)
    return


co_settings = _get_set()
fig_num = sys.argv[0].split('.py')[0].split('_')[-1]

ax_fs = co_settings['fig_settings']['ax_fs']
all_mask, arI, area_dat = _get_aux_data(co_settings)

color_list = co_settings[co_settings['fig_settings']['eval_region']]['colors']
color_list.insert(0, 'white')
#-------------------------------------------------------------------------------------------
# map of aridity classes from c2014 precip and '+pet_product+' pet (hyperarid is merged with arid)
_bounds_tmp = [0, 0.2, 0.5, 0.65, 1000000000000]
_tickLabs = [
    'Arid\n[<0.2]', 'Semi-arid\n[0.2-0.5]', 'Sub-humid\n[0.5-0.65]',
    'Humid\n[>0.65]'
]
# color_list = ["white", '#F44336', '#CCBB22', '#00BB77', '#1E88E5']
svname = 'fig_' + fig_num + co_settings['exp_suffix']
mk_glo_map(arI, _bounds_tmp)
