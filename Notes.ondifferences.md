   differences in different data cubes for tau...
   ```json
   "1ma":{
        "carv_name": "WWW",
        "fan_depth": 0,
        "comment": "fan 1m, carvalhais 2014 1m with only HWSD"
    },
    "1mb":{
        "carv_name": "WWWandNCSCD",
        "fan_depth": 0,
        "comment": "fan 1m, carvalhais 2014 1m with HWSD and NCSCD for northern hemisphere"

    },
    "fd":{
        "carv_name": "FullDepth",
        "fan_depth": 2,
        "comment": "fan full, carvalhais 2014 full depth"
    }
    ```


    to draw the figures
    bash make_all_figs.sh settings_common_macadamia.json 


    for the main text, settings_obs_tau_extended_cube_fd_extnlg.json is used as the settings. this is the cube without landgis. for supplement with soilgrids, the files with _extnsg is used. The fd means cSoil is calculated until the full depth.


to get the fraction of uncertainty agreement masks of Figure 4, run it using 
    bash make_all_figs.sh settings_common_macadamia.json which just runs this program, and the numbers are printed in the screen.



To calculate different variants of temperature, tas, tasp, taspn, taspneg5, slice_timeseries_and_calc_mean_raw_ts.py is used.
tas: the mean of all months of temperature
tasp: mean of all months with temperature <0 masked with nan
taspn: mean of all months with temperature <0 masked with zero
taspneg5 : mean of all months with temperature <-5 masked with nan

For now, taspneg5 figures are not drawn
figure with the following ending in file names:
no ending: uses tas
b: uses tasp
c: uses taspn

