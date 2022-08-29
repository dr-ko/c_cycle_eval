import numpy as np


def _get_regional_clim(_dat, area_dat, region_mask, co_settings):
    """
    return the median, and the percentiles of climate variables within a region
    """
    regn_list = np.unique(np.ma.masked_invalid(region_mask).compressed())
    __dat = np.zeros((3, len(regn_list) + 1))
    _perc_range = co_settings[co_settings['fig_settings']['eval_region']]['perc_range']

    for reg_num in range(len(regn_list)):
        _bID = int(regn_list[reg_num])
        biomSel = np.ma.getmask(np.ma.masked_equal(region_mask, _bID))

        _datZone = _dat[biomSel]
        __dat[0, reg_num + 1] = np.nanmedian(_datZone)
        __dat[1, reg_num + 1] = np.nanpercentile(_datZone, _perc_range[0])
        __dat[2, reg_num + 1] = np.nanpercentile(_datZone, _perc_range[1])
    biomSel = np.ma.getmask(np.ma.masked_not_equal(region_mask, np.nan))
    _datZone = _dat[biomSel]
    __dat[0, 0] = np.nanmedian(_datZone)
    __dat[1, 0] = np.nanpercentile(_datZone, _perc_range[0])
    __dat[2, 0] = np.nanpercentile(_datZone, _perc_range[1])
    return __dat

def _get_regional_means(_variable, _dat, area_dat, region_mask, co_settings):

    regn_list = np.unique(np.ma.masked_invalid(region_mask).compressed())
    __dat = np.zeros((len(regn_list) + 1))

    for reg_num in range(len(regn_list)):
        _bID = int(regn_list[reg_num])
        biomSel = np.ma.getmask(np.ma.masked_equal(region_mask, _bID))

        _datZone = _dat[biomSel]
        if _variable in co_settings['pgc_vars']:
            _areaZone = area_dat[biomSel]
            _datZone = _datZone * _areaZone
            __dat[reg_num + 1] = np.nansum(_datZone) * 1e-12
        else:
            __dat[reg_num + 1] = np.nanmedian(_datZone)
    biomSel = np.ma.getmask(np.ma.masked_not_equal(region_mask, np.nan))
    _areaZone = area_dat[biomSel]
    _datZone = _dat[biomSel]
    _datZone = _datZone * _areaZone
    __dat[0] = np.nansum(_datZone) * 1e-12
    return __dat


def _get_regional_range(_variable, _obsDataFull, area_dat, region_mask, co_settings):
    _perc_range = co_settings[co_settings['fig_settings']['eval_region']]['perc_range']
    nMemb = len(_obsDataFull)
    regn_list = co_settings[co_settings['fig_settings']['eval_region']]['regions']
    n_regions = len(regn_list)
    _perFull = np.zeros((n_regions + 1, nMemb))
    for memb in range(nMemb):
        zoneMemb = _get_regional_means(_variable, _obsDataFull[memb], area_dat, region_mask, co_settings)
        _perFull[:, memb] = zoneMemb[:]
    dat_low = np.nanpercentile(_perFull, _perc_range[0], axis=1)
    dat_high = np.nanpercentile(_perFull, _perc_range[1], axis=1)
    mod_dat_mean = np.median(_perFull, axis=1)
    regionalRange = np.zeros((2, len(dat_low)))
    regionalRange[0] = dat_low
    regionalRange[1] = dat_high
    return regionalRange, mod_dat_mean


def _get_regional_range_perFirst(_variable, _obsDataFull, area_dat, region_mask, co_settings):
    _perc_range = co_settings[co_settings['fig_settings']['eval_region']]['perc_range']
    dat_low = np.nanpercentile(_obsDataFull, _perc_range[0], axis=0)
    dat_high = np.nanpercentile(_obsDataFull, _perc_range[1], axis=0)
    mod_dat_zonal_5 = _get_regional_means(_variable, dat_low, area_dat, region_mask, co_settings)
    mod_dat_zonal_95 = _get_regional_means(_variable, dat_high, area_dat, region_mask, co_settings)
    regionalRange = np.zeros((2, len(mod_dat_zonal_5[:])))
    regionalRange[0] = mod_dat_zonal_5[:]
    regionalRange[1] = mod_dat_zonal_95[:]
    return regionalRange


def _get_regional_tau_c(_datgpp, _datc_total, area_dat, region_mask, co_settings):
    regn_list = np.unique(np.ma.masked_invalid(region_mask).compressed())
    __dat = np.zeros((len(regn_list) + 1))
    for reg_num in range(len(regn_list)):
        _bID = int(regn_list[reg_num])
        biomSel = np.ma.getmask(np.ma.masked_equal(region_mask, _bID))
        _datZonegpp = _datgpp[biomSel]
        _datZonec_total = _datc_total[biomSel]
        _areaZone = area_dat[biomSel]
        _datZonegpp = _datZonegpp * _areaZone
        _datZonec_total = _datZonec_total * _areaZone
        __dat[reg_num + 1] = np.nansum(_datZonec_total) / np.nansum(_datZonegpp)

    biomSel = np.ma.getmask(np.ma.masked_not_equal(region_mask, np.nan))
    _datZonegpp = _datgpp[biomSel]
    _datZonec_total = _datc_total[biomSel]
    _areaZone = area_dat[biomSel]
    _datZonegpp = _datZonegpp * _areaZone
    _datZonec_total = _datZonec_total * _areaZone
    __dat[0] = np.nansum(_datZonec_total) / np.nansum(_datZonegpp)
    return __dat


def _get_regional_tau_c_range(_obsgppFull, _obsc_totalFull, area_dat, region_mask, co_settings):
    _perc_range = co_settings[co_settings['fig_settings']['eval_region']]['perc_range']
    nMemb_gpp = len(_obsgppFull)
    nMemb_c_total = len(_obsc_totalFull)
    nMemb = nMemb_gpp * nMemb_c_total
    regn_list = np.unique(np.ma.masked_invalid(region_mask).compressed())
    n_regions = len(regn_list)
    _perFull = np.zeros((n_regions + 1, nMemb))
    memb_index = 0
    for memb_gpp in range(nMemb_gpp):
        gpp_memb = _obsgppFull[memb_gpp]
        for memb_c_total in range(nMemb_c_total):
            c_total_memb = _obsc_totalFull[memb_c_total]
            zoneMemb = _get_regional_tau_c(gpp_memb, c_total_memb, area_dat, region_mask, co_settings)
            _perFull[:, memb_index] = zoneMemb[:]
            memb_index = memb_index + 1
    dat_low = np.nanpercentile(_perFull, _perc_range[0], axis=1)
    dat_high = np.nanpercentile(_perFull, _perc_range[1], axis=1)
    mod_dat_mean = np.median(_perFull, axis=1)
    regionalRange = np.zeros((2, len(dat_low)))
    regionalRange[0] = dat_low
    regionalRange[1] = dat_high
    return regionalRange, mod_dat_mean
