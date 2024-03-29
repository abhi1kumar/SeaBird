"""
    Converts kitti image txt to kitti_360 window based npy and run evaluation

    Sample Run:
    python test/convert_and_evaluate_on_windows.py --split validation --result_folder output/gup_39/result_kitti_360/data
    python test/convert_and_evaluate_on_windows.py --split validation --result_folder data/kitti_360/validation/label/ --threshold 1
    python test/convert_and_evaluate_on_windows.py --split testing    --result_folder output/run_528/result_50_test_det/data
    python test/convert_and_evaluate_on_windows.py --split testing    --result_folder output/run_528/result_50_test_det/data --threshold 1 --replace_box
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

import argparse
from panoptic_bev.helpers.kitti_360_util import convert_kitti_image_text_to_kitti_360_window_npy
from panoptic_bev.helpers.kitti_360_evalDetection_windows import evaluate_kitti_360_windows_verbose

# ==================================================================================================
# Main starts here
# ==================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--split'        , default= 'validation', help= 'which split to use (validation/testing)')
parser.add_argument('--result_folder', default= 'output/gup_39/result_kitti_360/data', help='input folder name')
parser.add_argument('--threshold'    , type= float, default= 1.0, help='thrshold on distance')
parser.add_argument('--replace_box'  , action= 'store_true', default= False, help='replace low score box')
parser.parse_args()
args = parser.parse_args()

result_folder = args.result_folder
split         = args.split
max_dist_th   = args.threshold
replace_low_score_box = args.replace_box

# for replace_low_score_box in [False, True]:
#     for max_dist_th in [1.0, 4.]:
for replace_low_score_box in [True]:
    for max_dist_th in [4.]:
        if max_dist_th == 4 and not replace_low_score_box:
            continue
        convert_kitti_image_text_to_kitti_360_window_npy(result_folder, split= split, max_dist_th= max_dist_th,
                                                         replace_low_score_box= replace_low_score_box, logger= None, verbose= False)

        if args.split != 'testing':
            # Run evaluation on windows
            if "data/kitti_360/" in result_folder:
                output_folder = os.path.join("output/oracle_img_to_win", "{}_th_{:.0f}_replace_{}".format(split, max_dist_th, str(replace_low_score_box)))
            else:
                output_folder = result_folder.replace("data", "data_kitti_360_format")

            evaluate_kitti_360_windows_verbose(pred_folder= output_folder, gt_folder= "data/kitti_360/train_val/windows/")

            # command = "python src/helpers/kitti_360_evalDetection_windows.py " + output_folder
            # os.system(command)
