import glob, os
import os
import xarray as xr
import numpy as np
overWriteData = False
# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"
mainDir_t='/Net/Groups/BGI/work_3/FluxcomDataStructure/internal/QOVbZtjf6sSIZowqIc99/CarbonFluxes/RS/0d50/8daily/'
# /Net/Groups/BGI/work_3/FluxcomDataStructure/internal/QOVbZtjf6sSIZowqIc99/CarbonFluxes/RS/0d50/8daily
dat = 'ANNnoPFT  GMDH_CV  KRR  MARSens  MTE  MTEM  MTE_Viterbo  RFmiss  SVM'.split()
odir = './fluxcom_all_rs_memb'
os.makedirs(odir, exist_ok=True)
for _dat in dat:
    mainDir = os.path.join(mainDir_t, _dat)
    for root, dirs, files in os.walk(mainDir):
        for file in files:
            # print (file)
            if file.startswith("GPP") and file.endswith(".nc") and 'ltMean' not in file:
                yr = int(file.split('.')[-2])
                # if yr == 1982:
                if yr > 2000 and yr < 2016:
                    infile=os.path.join(root, file)
                    ofilename = file.replace('.nc','.rs-ltMean.nc')
                    print (infile)
                    indat=xr.open_mfdataset(infile)
                    if file.startswith('GPP_HB'):
                        tmp=indat['GPP_HB'].values
                    else:
                        tmp=indat['GPP'].values
                    tmp[tmp<0]=0
                    tmp_mean_masked = np.nansum(tmp,0) * (8.0 * 365.25)/(1000 * 368.)
                    dat_mean = indat.mean(dim='time')
                    if file.startswith('GPP_HB'):
                        dat_mean['GPP_HB'].values = tmp_mean_masked
                    else:
                        dat_mean['GPP'].values = tmp_mean_masked
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