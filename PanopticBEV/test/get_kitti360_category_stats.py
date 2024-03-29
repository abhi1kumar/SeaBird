"""
    Sample Run:
    python test/get_kitti360_category_stats.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

import glob
from src.helpers.file_io import imread, read_csv

kitti_360_folder  = "data/kitti_360/validation"
label_folder      = os.path.join(kitti_360_folder, "label")

list_of_label_files  = sorted(glob.glob(label_folder + "/*.txt"))

dimension_list = [None, None, None, None]
class_list_name= ['Car', 'Cyclist', 'Pedestrian', 'Building']
cat_count_list = [0, 0, 0, 0]

for i, label_file_path in enumerate(list_of_label_files):
    label_gt        = read_csv(label_file_path, ignore_warnings= True, use_pandas= True)

    if ((i+1) % 1000 ==0) or (i+1) == len(list_of_label_files):
        print("{} images done".format(i+1))
    if label_gt is not None:
        # Filter out cars first
        for k, class_name in enumerate(class_list_name):
            class_gt   = label_gt[label_gt[:, 0] == class_name]
            num_boxes  = class_gt.shape[0]

            if num_boxes <= 0:
                continue

            data_boxes = class_gt[:, 1:].astype(np.float32)

            #       0  1   2      3   4   5   6    7    8    9   10   11   12   13
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d
            alpha = data_boxes[:, 2]
            x1  = data_boxes[:, 3]
            y1  = data_boxes[:, 4]
            x2  = data_boxes[:, 5]
            y2  = data_boxes[:, 6]
            h3d = data_boxes[:, 7]
            w3d = data_boxes[:, 8]
            l3d = data_boxes[:, 9]
            x3d = data_boxes[:, 10]
            y3d = data_boxes[:, 11]
            z3d = data_boxes[:, 12]
            ry3d = data_boxes[:,13]

            if dimension_list[k] is None:
                dimension_list[k] = data_boxes[:, 7:10]
            else:
                dimension_list[k] = np.vstack((dimension_list[k], data_boxes[:,7:10]))
            cat_count_list[k] += num_boxes

for k, dimension_class in enumerate(dimension_list):
    comp_arr = np.array(dimension_class)
    print(class_list_name[k], cat_count_list[k], np.mean(comp_arr, axis= 0))
