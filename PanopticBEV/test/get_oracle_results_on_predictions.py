"""
    Sample Run:
    python test/get_oracle_results_on_predictions.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

from src.helpers.seman_helper import run_oracle_on_predictions

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

label_folder     = "data/kitti_360/validation/label"
result_folder    = "output/run_644/result_40_val_det_samp_conf_0_1/data"
list_of_cat      = ['Car', 'Building']
update_cat       = []
dist_max_th_list = {'Car': 4, 'Building': 14}
oracle_input_list= ['h3d', 'y3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

update_cat       = ['Car', 'Building']
# oracle_input_list= ['h3d', 'y3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat)

# oracle_input_list= ['w3d', 'z3d', 'l3d', 'x3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat)


# dist_max_th_list   = {'Car': 1, 'Building': 7}
# oracle_input_list  = ['h3d', 'w3d', 'l3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# dist_max_th_list   = {'Car': 2, 'Building': 14}
# oracle_input_list  = ['h3d', 'w3d', 'l3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

#
# dist_max_th_list   = {'Car': 4, 'Building': 28}
# oracle_input_list  = ['y3d', 'z3d', 'x3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)


# dist_max_th_list   = {'Car': 4, 'Building': 28}
# oracle_input_list  = ['y3d', 'z3d', 'x3d', 'score']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# dist_max_th_list   = {'Car': 1, 'Building': 7}
# oracle_input_list  = ['y3d', 'z3d', 'x3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# dist_max_th_list   = {'Car': 1, 'Building': 7}
# oracle_input_list  = ['y3d', 'z3d', 'x3d', 'score']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# dist_max_th_list   = {'Car': 4, 'Building': 14}
# oracle_input_list  = ['y3d', 'z3d', 'x3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 14}
# oracle_input_list  = ['y3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 14}
# oracle_input_list  = ['z3d', 'x3d']
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# ==================================================================================================
# Ablation on different thresholds with centers
# ==================================================================================================
# oracle_input_list  = ['x3d', 'z3d', 'y3d']
# dist_max_th_list   = {'Car': 1, 'Building': 7}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 4}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 14}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

# ==================================================================================================
# Ablation on different thresholds with BEV centers
# ==================================================================================================
# oracle_input_list  = ['x3d', 'z3d']
# dist_max_th_list   = {'Car': 1, 'Building': 7}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 4}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
#
# dist_max_th_list   = {'Car': 4, 'Building': 14}
# run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)



dist_max_th_list   = {'Car': 4, 'Building': 14}
"""
oracle_input_list  = ['x3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['z3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['y3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d', 'y3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d', 'y3d', 'h3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['ry3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d', 'y3d', 'h3d', 'ry3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
"""
oracle_input_list  = ['l3d', 'w3d', 'h3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d', 'y3d', 'l3d', 'w3d', 'h3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)

oracle_input_list  = ['x3d', 'z3d', 'y3d', 'l3d', 'w3d', 'h3d', 'ry3d']
run_oracle_on_predictions(label_folder, result_folder, oracle_input_list= oracle_input_list, list_of_cat= list_of_cat, update_cat= update_cat, dist_max_th_list= dist_max_th_list)
