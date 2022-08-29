import cartopy.crs as ccrs
import os
import xarray as xr
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['hatch.color'] = '#888888'
plt.rcParams.update({'font.size':8})

from _shared import _get_set, _get_data, _rem_nan, _apply_a_mask
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


#--------
def get_pet_data(_whichVar, all_mask, co_settings):
    if _whichVar == 'modis':
        datfile=os.path.join(co_settings['top_dir'],'Koven2014/datasets/MOD16A3_Science_PET_mean_00_13_regridhalfdegree.nc')
        datVar='pet'
        mod_dat_f=xr.open_dataset(datfile,decode_times=False)
        mod_dat0=mod_dat_f[datVar].values.reshape(-1,360,720).mean(0)
        mod_dat0=np.flipud(mod_dat0)
    if _whichVar == 'cgiar':
        datfile=os.path.join(co_settings['top_dir'],'Aridity/et0_yr/et0_yr_0d5.tif')
        datVar='pet'
        mod_dat_f=xr.open_rasterio(datfile)
        print(mod_dat_f)
        mod_dat0=mod_dat_f.values.reshape(-1,300,720).mean(0)
        mod_dat0[mod_dat0<0]=np.nan
        omod = np.ones((360,720))*np.nan
        omod[0:300,:]=mod_dat0
        mod_dat0=omod
    mod_dat=_rem_nan(mod_dat0)
    mod_dat=_apply_a_mask(mod_dat,all_mask)
    return mod_dat

def get_kov_pr(co_settings):
    datfile=os.path.join(co_settings['top_dir'],'Koven2014/datasets/MAAP.360.720.1.nc')
    datVar='MAAP'
    datCorr=1.
    mod_dat_f=xr.open_dataset(datfile,decode_times=False)
    mod_dat0=mod_dat_f[datVar].values.reshape(-1,360,720).mean(0)#*
    return(mod_dat0)

def get_pr_bins(_dat,_bounds_tmp):
    _odat=np.ones(np.shape(_dat))
    for _tr in range(len(_bounds_tmp)-1):
        tas1=_bounds_tmp[_tr]
        tas2=_bounds_tmp[_tr+1]
        mod_tas_tmp=np.ma.masked_inside(_dat,tas1,tas2).mask
        _odat[mod_tas_tmp]=_tr+1
    _odat=_apply_a_mask(_odat,all_mask)
    return _odat

def mk_glo_map(datBas,_bounds_tmp):
    npoint=np.ma.count(np.ma.masked_invalid(datBas))

    cm2 = mpl.colors.ListedColormap(color_list) 
    bounds2 = np.arange(-0.5,len(_bounds_tmp),1)#([-0.5,0.5, 1.5,2.5,3.5,4.5])
    vegcls=_tickLabs
    vgMap=datBas
    #---------------------------
    plt.figure()
    _ax = plt.subplot(1,1,1, projection=ccrs.Robinson(central_longitude=0),
                   frameon=False)
    print(vgMap)
    plt.imshow(np.ma.masked_less(vgMap[:300,:],-999.), cmap=cm2, norm=mpl.colors.BoundaryNorm(bounds2,cm2.N),interpolation='none',origin='upper',
               transform=ccrs.PlateCarree(),
               extent=[-180, 180, -60, 90])
    _ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    _ax.coastlines(linewidth=0.4, color='grey')
    plt.gca().outline_patch.set_visible(False)
    cblab=[]
    for vg in range(len(vegcls)):
        lpvgt=np.ma.nonzero(np.ma.masked_not_equal(vgMap,vg+1).filled(0))
        pcon=str(np.round(len(lpvgt[0])*1./npoint*100.,1))+"%"
        cbtxt=vegcls[vg]+'\n('+pcon+')'
        cblab=np.append(cblab,cbtxt)
        print (vegcls[vg],cbtxt)
    cb=plt.colorbar(boundaries=bounds2[1:],shrink=.75,aspect=50,drawedges=True,orientation='horizontal',pad=0.01)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(7.52)
        t.set_rotation(0)
    #    cb.set_ticks(arange(colomin,colomax+coloint,coloint))
    ylocss=[]
    for jp in cb.ax.xaxis.get_ticklocs():
        print (jp)
        ylocss=np.append(ylocss,jp)
    ylocar=[]
    for _yy in range(len(ylocss)-1):
        ylocar=np.append(ylocar,0.5*(ylocss[_yy]+ylocss[_yy+1]))
        print ("haha")
    cb.ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ylocar))
    #    cb.ax.yaxis.set_tickloc(FixedLocator(arange(1,6)))
    cb.ax.set_xticklabels(cblab)

    fix_cb(cb,ax_fs=ax_fs)
    t1 = plt.figtext(-0.04,0.30,' ',transform=plt.gca().transAxes)
    t2 = plt.figtext(1.03,1.06,' ',transform=plt.gca().transAxes)
    plt.savefig(os.path.join(figs_dir, svname+'.'+ co_settings['fig_settings']['fig_format']),bbox_inches='tight',bbox_extra_artists=[t1,t2],dpi=450)
    plt.close(1)

def fix_cb(_cb,ax_fs=6):
    _cb.ax.tick_params(labelsize=ax_fs,size=2,width=0.3)
    print (_cb.ax.yaxis.get_ticklocs())
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
    cblabs=cb.ax.yaxis.get_ticklabels()
    xlabs=[]
    for __b in cblabs:
        _b=__b.get_text()
        # print(float(_b))
        if _b == '0':
            xlabs=np.append(xlabs,_b)
        else:
            xlabs=np.append(xlabs,'%d'%(round(np.power(10,float(_b)),0)))

    cb.ax.set_yticklabels(xlabs)
    return


co_settings = _get_set()

top_dir = co_settings['top_dir']
obs_dict = co_settings['obs_dict']

# add observation to models list
models_only = co_settings['model']['names']
models_only.insert(0, 'obs')
co_settings['model']['names'] = models_only
models = co_settings['model']['names']
model_dict = co_settings['model_dict']
nmodels = len(models)

masks_dir = os.path.join(top_dir, './masks_n/_data/')
os.makedirs(masks_dir, exist_ok=True)
figs_dir = os.path.join(top_dir, './masks_n/_maps/')
os.makedirs(figs_dir, exist_ok=True)

ax_fs=co_settings['fig_settings']['ax_fs']


mask_all='none model_valid model_valid_tau_cObs model_valid_tau_cObs_KGBW model_valid_tau_cObs_gppObsgt1GC model_valid_tau_cObs_gppObsgt10GC model_valid_tau_cObs_gppObsgt100GC model_valid_tau_cObs_prObsgt50mm model_valid_tau_cObs_prObsgt100mm model_valid_tau_cObs_prObsgt250mm'.split()
for _whichMask in mask_all:
    all_mask = xr.open_dataset(os.path.join(masks_dir, 'mask_'+_whichMask+'.nc'))['mask'].values.astype(int)
    all_mod_dat_tas = _get_data('tas', co_settings, _co_mask=all_mask)
    all_mod_dat_pr = _get_data('pr', co_settings, _co_mask=all_mask)

    #---------------------------------------------------------------------------
    # save the maps of classes based on precipitation bins
    #---------------------------------------------------------------------------
    _tickLabs=['<500','500-1000','1000-2000','>2000']
    _bounds_tmp=[0,500,1000,2000,5000]
    color_list=["white",'#F44336','#FDD835','#00ff33','#1E88E5']
    pr_obs=all_mod_dat_pr['obs']
    pr_mod=all_mod_dat_pr['lpj']
    pr_obs_bin=get_pr_bins(pr_obs,_bounds_tmp)
    pr_mod_bin=get_pr_bins(pr_mod,_bounds_tmp)

    create_nc_hlf(os.path.join(masks_dir, 'map_pr_bins_mod_'+_whichMask+'.nc'), 'mask', pr_mod_bin.astype(np.float32))


    create_nc_hlf(os.path.join(masks_dir, 'map_pr_bins_obs_'+_whichMask+'.nc'), 'mask', pr_obs_bin.astype(np.float32))

    svname='map_pr_bins_obs_'+_whichMask
    mk_glo_map(pr_obs_bin,_bounds_tmp)
    svname='map_pr_bins_mod_'+_whichMask
    mk_glo_map(pr_mod_bin,_bounds_tmp)

    #---------------------------------------------------------------------------
    # save the maps of classes based on temperature bins
    #---------------------------------------------------------------------------
    color_list=["white",'purple','#1E88E5','#00ff33','#FDD835','#F44336']
    color_list=["white",'#1E88E5','#00ff33','#FDD835','#F44336']
    _tickLabs=['<5','5-15','15-25','>25']
    _bounds_tmp=[-100,5,15,25,100]

    tas_obs=all_mod_dat_tas['obs']
    tas_mod=all_mod_dat_tas['lpj']
    tas_obs_bin=get_pr_bins(tas_obs,_bounds_tmp)
    tas_mod_bin=get_pr_bins(tas_mod,_bounds_tmp)

    create_nc_hlf(os.path.join(masks_dir, 'map_tas_bins_mod_'+_whichMask+'.nc'), 'mask', tas_mod_bin.astype(np.float32))


    create_nc_hlf(os.path.join(masks_dir, 'map_tas_bins_obs_'+_whichMask+'.nc'), 'mask', tas_obs_bin.astype(np.float32))

    svname='map_tas_bins_obs_'+_whichMask
    mk_glo_map(tas_obs_bin,_bounds_tmp)
    svname='map_tas_bins_mod_'+_whichMask
    mk_glo_map(tas_mod_bin,_bounds_tmp)

    #---------------------------------------------------------------------------
    # save the maps of classes based on aridity and pet
    #---------------------------------------------------------------------------

    for pet_product in ['cgiar','modis']:

        all_mod_dat_pet = get_pet_data(pet_product, all_mask, co_settings)

        #-------------------------------------------------------------------------------------------
        # map of the pr-pet from C2014 precip and '+pet_product+' pet
        _tickLabs=['<-3000','-3000 to\n-2000','-2000 to\n-1000','-1000 to\n0','0 to\n1000','>1000']
        _bounds_tmp=[-100000,-3000,-2000,-1000,0,1000,10000]
        color_list=["white",'purple','#F44336','#FDD835','#00ff33','#1E88E5','black']
        pr_obs=all_mod_dat_pr['obs']
        pr_pet_obs=all_mod_dat_pr['obs']-all_mod_dat_pet
        pr_obs_bin=get_pr_bins(pr_pet_obs,_bounds_tmp)
        create_nc_hlf(os.path.join(masks_dir, 'mask_pr-pet-'+pet_product+'_obs_'+_whichMask+'.nc'), 'mask', pr_obs_bin.astype(np.float32))
        svname='map_pr-pet_bins_obs_'+_whichMask+''
        mk_glo_map(pr_obs_bin,_bounds_tmp)

        #-------------------------------------------------------------------------------------------
        # mask of the pr-pet < -1000 from '+pet_product+' precip and '+pet_product+' pet
        color_list=['#FFFFFF','#F44336','#1E88E5']
        _tickLabs=['excl','incl']
        _bounds_tmp=[-10000,-1000,10000]
        kov_pr=get_kov_pr(co_settings)
        pr_pet_kov=kov_pr-all_mod_dat_pet
        pr_pet_kov_bin=get_pr_bins(pr_pet_kov,_bounds_tmp)
        print('kkkkkk',pr_pet_kov)
        create_nc_hlf(os.path.join(masks_dir, 'mask_ari-'+pet_product+'_obs_'+_whichMask+'.nc'), 'mask', pr_pet_kov_bin.astype(np.float32))
        svname='mask_koven2014_obs_'+_whichMask+''
        mk_glo_map(pr_pet_kov_bin,_bounds_tmp)

        #-------------------------------------------------------------------------------------------
        # mask of the pr-pet < -1000 from crescendo precip and '+pet_product+' pet
        # color_list=['#F44336','#1E88E5']
        color_list=['#FFFFFF','#F44336','#1E88E5']
        _tickLabs=['excl','incl']
        _bounds_tmp=[-10000,-1000,10000]
        pr_mod=all_mod_dat_pr['lpj']
        pr_pet_kov=pr_mod-all_mod_dat_pet
        pr_pet_kov_bin=get_pr_bins(pr_pet_kov,_bounds_tmp)
        create_nc_hlf(os.path.join(masks_dir, 'mask_ari-'+pet_product+'_mod_'+_whichMask+'.nc'), 'mask', pr_pet_kov_bin.astype(np.float32))
        svname='mask_koven2014_mod_'+_whichMask+''
        mk_glo_map(pr_pet_kov_bin,_bounds_tmp)


        #-------------------------------------------------------------------------------------------
        # map of aridity classes from c2014 precip and '+pet_product+' pet (includes hyperarid)   
        _bounds_tmp=[0,0.03,0.2,0.5,0.65,1000]
        _tickLabs=['Hyper\nArid [<0.03]','Arid\n[0.03-0.2]','Semi-arid\n[0.2-0.5]','Sub-humid\n[0.5-0.65]','Humid\n[>0.65]']
        color_list=["white",'#F44336','#FDD835','#00ff33','#1E88E5','black']
        pr_obs=all_mod_dat_pr['obs']
        pet_obs=all_mod_dat_pet
        pet_obs[pet_obs < 0.]=0.1
        pr_pet_obs=pr_obs/pet_obs
        pr_obs_bin=get_pr_bins(pr_pet_obs,_bounds_tmp)
        svname='map_ari-'+pet_product+'_bins_obs_'+_whichMask+''
        mk_glo_map(pr_obs_bin,_bounds_tmp)

        #-------------------------------------------------------------------------------------------
        # map of aridity classes from c2014 precip and '+pet_product+' pet (hyperarid is merged with arid)   
        _bounds_tmp=[0,0.2,0.5,0.65,1000000000000]
        _tickLabs=['Arid\n[<0.2]','Semi-arid\n[0.2-0.5]','Sub-humid\n[0.5-0.65]','Humid\n[>0.65]']
        color_list=["white",'#F44336','#CCBB22','#00BB77','#1E88E5']

        pr_obs=all_mod_dat_pr['obs']
        pet_obs=all_mod_dat_pet
        pet_obs[pet_obs <= 0.]=0.0001
        pr_pet_obs=pr_obs/pet_obs
        pr_obs_bin=get_pr_bins(pr_pet_obs,_bounds_tmp)
        create_nc_hlf(os.path.join(masks_dir, 'map_ari-'+pet_product+'_4c_bins_obs_'+_whichMask+'.nc'), 'aridity_class', pr_obs_bin.astype(np.float32))
        svname='map_ari-'+pet_product+'_4c_bins_obs_'+_whichMask+''
        mk_glo_map(pr_obs_bin,_bounds_tmp)

        #-------------------------------------------------------------------------------------------
        # map of aridity classes from crescendo precip and '+pet_product+' pet (hyperarid is merged with arid)   
        pr_obs=all_mod_dat_pr['lpj']
        pet_obs=all_mod_dat_pet

        pet_obs[pet_obs <= 0.]=0.0001
        pr_pet_obs=pr_obs/pet_obs
        pr_obs_bin=get_pr_bins(pr_pet_obs,_bounds_tmp)
        create_nc_hlf(os.path.join(masks_dir, 'map_ari-'+pet_product+'_4c_bins_mod_'+_whichMask+'.nc'), 'aridity_class', pr_obs_bin.astype(np.float32))
        svname='map_ari-'+pet_product+'_4c_bins_mod_'+_whichMask+''
        mk_glo_map(pr_obs_bin,_bounds_tmp)
