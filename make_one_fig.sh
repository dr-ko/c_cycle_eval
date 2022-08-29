#!/bin/bash

export common_settings_cres=$1

# data settings
declare -a setting_array=("extended_cube_fd_extnlg" "extended_cube_fd_extnsg")

# programs to run
declare -a prog_array=("figure_supp_A7.py" "figure_supp_A12.py")

for val in "${prog_array[@]}"; do
  for setting in "${setting_array[@]}"; do
    echo ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    echo Running: $val
    c_cycle_obs_set=$setting /Net/Groups/BGI/people/skoirala/anaconda3/envs/cres/bin/python $val &
    echo Complete: $val
    echo ^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_^_
  done
done

