#!/bin/bash
# Declare a string array of programs to be run
export common_settings_cres=$1

declare -a setting_array=("extended_cube_fd_extnlg" "extended_cube_fd_extnsg")

declare -a prog_array=("figure_main_01.py" "figure_main_02.py" "figure_main_03.py" "figure_main_04.py" "figure_main_05.py" "figure_supp_A1.py" "figure_supp_A2.py" "figure_supp_A3.py" "figure_supp_A4.py"  "figure_supp_A5.py"  "figure_supp_A6.py" "figure_supp_A8-A9-A10.py" "figure_supp_A11.py" "figure_supp_A12.py" "figure_supp_A13.py")

for val in "${prog_array[@]}"; do
  for setting in "${setting_array[@]}"; do
  # export c_cycle_obs_set=$setting
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Running: $val
    c_cycle_obs_set=$setting /Net/Groups/BGI/people/skoirala/anaconda3/envs/cres/bin/python $val &
    echo Complete: $val
    echo ^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_
  done
done
