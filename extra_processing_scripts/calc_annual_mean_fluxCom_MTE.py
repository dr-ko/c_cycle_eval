import glob, os
import os
import xarray as xr
import numpy as np
overWriteData = False
# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"
mainDir_t='/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d50_monthly/'
# dat = 'CERES_GPCP  CRUNCEP_crujra_v1.1  CRUNCEP_v6  CRUNCEP_v8  GSWP3  WFDEI  era5'.split()
dat = 'CRUNCEP_crujra_v1.1  CRUNCEP_v8  GSWP3  era5'.split()
dat = 'EnsembleGPP_GL EnsembleGPP_MR'.split()
odir = './fluxcom_all_mte_memb'
os.makedirs(odir, exist_ok=True)
for _dat in dat:
    mainDir = os.path.join(mainDir_t, _dat + '/May12/Data/')
    for root, dirs, files in os.walk(mainDir):
        for file in files:
            # print (file)
            if file.startswith("Ensemble") and file.endswith(".nc") and 'ltMean' not in file:
                yr = int(file.split('.')[-2])
                # if yr == 1982:
                if yr > 1981 and yr < 2012:
                    infile=os.path.join(root, file)
                    ofilename = file.replace('.nc','.mte-ltMean.nc')
                    print (infile)
                    indat=xr.open_mfdataset(infile, decode_times=False)
                    tmp=indat[_dat+'_May12'].values
                    tmp[tmp<0]=np.nan
                    tmp_mean_masked = np.nanmean(tmp,0) * 365.25/ 1000.
                    dat_mean = indat.mean(dim='time')
                    dat_mean[_dat+'_May12'].values = tmp_mean_masked
                    ofile = os.path.join(odir, ofilename)
                    dat_mean.to_netcdf(ofile)
                    print(ofile)
                    indat.close()
                    print('------------------------')
                # ofile=infile.replace('.nc','.ltMean.nc')
                # # ofile=infile.replace('.nc','.ltMean')
                # print(infile,ofile)
                # if os.path.exists(ofile) == False:
                #     dat_=xr.open_dataset(infile,decode_times=False)
                #     dat_mean=dat_.mean(dim='time',skipna=True)
                #     print(dat_,dat_mean)
                #     dat_mean.to_netcdf(path=ofile, mode='w')
                # elif overWriteData == True:
                #     dat_=xr.open_dataset(infile,decode_times=False)
                #     dat_mean=dat_.mean(dim='time',skipna=True)
                #     print(dat_,dat_mean)
                #     dat_mean.to_netcdf(path=ofile, mode='w')