import glob, os
import os
import xarray as xr
import numpy as np
overWriteData = False
# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"
dat = 'CERES_GPCP  CRUNCEP_crujra_v1.1  CRUNCEP_v6  CRUNCEP_v8  GSWP3  WFDEI  era5'.split()
# dat = 'CRUNCEP_crujra_v1.1  CRUNCEP_v8  GSWP3  era5'.split()
mlms = 'RF ANN MARS'.split()
varbs = 'GPP_HB GPP'.split()


mainDir_t = './fluxcom_all_rsm_memb'
odir = './fluxcom_all_rsm_memb/long_term_mean'
os.makedirs(odir, exist_ok=True)
for _dat in dat:
    if _dat == 'CERES_GPCP':
        syr=2001
    else:
        syr=2001
    if _dat == 'GSWP3':
        eyr=2010
    else:
        eyr=2010
    nYrs = eyr - syr + 1
    mainDir = os.path.join(mainDir_t, '')
    for _mlm in mlms:
        for _varb in varbs:
            ofile_n = '{_varbi}.{_mlmi}.{_dati}.daily.rsm-long_term_mean.nc'.format(_varbi=_varb, _mlmi=_mlm, _dati=_dat)
            ofile = os.path.join(odir, ofile_n)
            odat = np.ones((nYrs, 360,720))
            yr_ind = 0
            for _yr in range(syr, eyr+1):
                # print (file)
                infile_n = '{_varbi}.{_mlmi}.{_dati}.daily.{_yri}.rsm-ltMean.nc'.format(_varbi=_varb, _mlmi=_mlm, _dati=_dat, _yri=str(_yr))
                infile=os.path.join(mainDir, infile_n)
                # ofilename = file.replace('.nc','.rsm-ltMean.nc')
                print (infile)
                indat=xr.open_dataset(infile)
 
                if _varb == 'GPP_HB':
                    indat = indat.rename({'GPP_HB':'GPP'})
                #     print(indat)
                datv = indat['GPP'].values
                print(_varb,np.nanmax(datv),np.nanmin(datv))
                odat[yr_ind]=datv
                yr_ind = yr_ind + 1
                if _yr < eyr:
                    indat.close()
            indat['GPP'].values = np.nanmean(odat,0)
            indat.to_netcdf(ofile)
            print(ofile, ' ------saved')
            print('-------------------------------')