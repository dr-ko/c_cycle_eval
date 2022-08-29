import glob, os
import os
import xarray as xr
import numpy as np
overWriteData = False
# overWriteData = True

# mainDir="/media/skoirala/exStore/FLUXCOM_Eflux/analysis_eFlux_paper_iter2_201810/data.local/"

dat = 'EnsembleGPP_GL EnsembleGPP_MR'.split()

mainDir_t = './fluxcom_all_mte_memb'
odir = './fluxcom_all_mte_memb/long_term_mean'
os.makedirs(odir, exist_ok=True)
for _dat in dat:
    syr=2001
    eyr=2010
    nYrs = eyr - syr + 1
    mainDir = os.path.join(mainDir_t, '')
    # EnsembleGPP_GL_May12.1992.mte-ltMean.nc
    ofile_n = '{_dati}.mte-ltMean.nc'.format(_dati=_dat)
    ofile = os.path.join(odir, ofile_n)
    odat = np.ones((nYrs, 360,720))
    yr_ind = 0
    for _yr in range(syr, eyr+1):
        # print (file)
        infile_n = '{_dati}_May12.{_yri}.mte-ltMean.nc'.format(_dati=_dat, _yri=str(_yr))
        infile=os.path.join(mainDir, infile_n)
        # ofilename = file.replace('.nc','.mte-ltMean.nc')
        print (infile)
        indat=xr.open_dataset(infile, decode_times=False)

        indat = indat.rename({_dat+'_May12':'GPP'})
        #     print(indat)
        datv = indat['GPP'].values
        print(np.nanmax(datv),np.nanmin(datv))
        odat[yr_ind]=datv
        yr_ind = yr_ind + 1
        if _yr < eyr:
            indat.close()
    indat['GPP'].values = np.nanmean(odat,0)
    indat.to_netcdf(ofile)
    print(ofile, ' ------saved')
    print('-------------------------------')