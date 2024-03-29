"""
    Sample Run:
    python test/get_lengthwise_analysis.py --result_folder output/run_645/result_40_val_det_samp/data_kitti_360_format

    Run lengthwise analysis on the KITTI-360 detection result.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import argparse

np.set_printoptions   (precision= 4, suppress= True)
from panoptic_bev.helpers.kitti_360_evalDetection_windows import evaluate_kitti_360_windows_verbose

# ==================================================================================================
# Main starts here
# ==================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--result_folder', default= 'output/run_645/result_40_val_det_samp/data_kitti_360_format', help='input folder name')
parser.parse_args()
args = parser.parse_args()


length_max     = 20
num_bins       = 4
output_folder  = args.result_folder

length_bins    = np.arange(0, length_max + 0.01, step= length_max / num_bins)
length_bins[0] = -1

for i in range(num_bins):
    dim_min = length_bins[i]
    dim_max = length_bins[i+1]
    print("\n==============================================================")
    print("Length Range [{:.2f}, {:.2f}]".format(dim_min, dim_max))
    print("==============================================================")
    evaluate_kitti_360_windows_verbose(pred_folder= output_folder, gt_folder= "data/kitti_360/train_val/windows/", dim_min= dim_min, dim_max= dim_max)


print("\n==============================================================")
print("Length All")
print("==============================================================")
evaluate_kitti_360_windows_verbose(pred_folder= output_folder, gt_folder= "data/kitti_360/train_val/windows/", dim_min= -1, dim_max= 10000)
