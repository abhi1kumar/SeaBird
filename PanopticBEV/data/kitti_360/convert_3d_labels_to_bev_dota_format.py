"""
    Sample Run:
    python data/kitti_360/convert_3d_labels_to_bev_dota_format.py

    Converts KITTI-360 labels to BEV in DOTA format.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import glob

from panoptic_bev.helpers.seman_helper import metric_to_bev_px, get_obj_level
from panoptic_bev.helpers.file_io import read_csv, write_lines

def convert_to_dota(label_folder):
    list_of_label_files = sorted(glob.glob(label_folder + "/*.txt"))
    if "label_2" in label_folder:
        output_label_folder = label_folder.replace("label_2", "label_2_dota")
    else:
        output_label_folder = label_folder.replace("label", "label_dota")
    print("\nLabel folder= {}".format(label_folder))
    print("Outpt folder= {}".format(output_label_folder))
    bev_w = 768
    bev_h = 704

    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    for index, label_file_path in enumerate(list_of_label_files):
        img_name        = os.path.basename(label_file_path).replace(".txt", "")

        label_gt        = read_csv(label_file_path, ignore_warnings= True, use_pandas= True)
        lines_out = []
        if label_gt is not None:
            # Filter out cars first
            relevant_index  = np.logical_or(label_gt[:, 0] == 'Car', label_gt[:, 0] == 'Building')
            label_gt   = label_gt[relevant_index]
            num_boxes  = label_gt.shape[0]
            class_boxes= label_gt[:, 0]
            data_boxes = label_gt[:, 1:].astype(np.float32)

            #       0  1   2      3   4   5   6    7    8    9   10   11   12   13
            # cls, trun, occ, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d
            truncation = data_boxes[:, 0]
            occlusion = data_boxes[:, 1]
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
            yaw_kitti = data_boxes[:,13]
            ry3d = - data_boxes[:,13] # KITTI convention is clockwise. Convert to anticlockwise
            height = y2 - y1

            # Rotate boxes around center
            rot_mat = np.zeros((num_boxes, 2, 2))
            rot_mat[:, 0, 0] =  np.cos(ry3d)
            rot_mat[:, 0, 1] = -np.sin(ry3d)
            rot_mat[:, 1, 0] =  np.sin(ry3d)
            rot_mat[:, 1, 1] =  np.cos(ry3d)

            points = np.zeros((num_boxes, 4, 2))
            #                 Z
            #                 |
            #  3(-l/2, w/2)   |        0 (l/2, w/2)
            #  ---------------|-------|
            # |               |       |
            # |               |-------|--> X
            # |                       |
            # |-----------------------|
            # 2 (-l/2, w/2)           1 (l/2, -w/2)

            points[:, 0, 0] += l3d/2.0
            points[:, 0, 1] += w3d/2.0
            points[:, 1, 0] += l3d/2.0
            points[:, 1, 1] -= w3d/2.0
            points[:, 2, 0] -= l3d/2.0
            points[:, 2, 1] -= w3d/2.0
            points[:, 3, 0] -= l3d/2.0
            points[:, 3, 1] += w3d/2.0

            # rotate points
            rotated_points = np.matmul(rot_mat, points.transpose(0, 2, 1)).transpose(0, 2, 1) # N x 4 x 2

            # now move the center
            rotated_points[:, :, 0] += x3d[:, np.newaxis]
            rotated_points[:, :, 1] += z3d[:, np.newaxis]

            rotated_points_px = metric_to_bev_px(center_metric= rotated_points, bev_h= bev_h, bev_w= bev_w)

            for j in np.arange(num_boxes):
                obj_level = get_obj_level(height[j], truncation[j], occlusion[j])
                if obj_level == 1 or obj_level == 2:
                    difficulty = 0
                else:
                    difficulty = 1
                # Format:         x1,   y1,    x2,    y2,    x3,    y3,    x4,    y4, cat, dif, alpha,   x1,    y1,    x2,    y2,   h3d,   w3d,   l3d,   x3d,   y3d,   z3d,   ry3d
                output_str = ("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {} {:1d} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n"
                    .format(rotated_points_px[j, 0, 0], rotated_points_px[j, 0, 1],
                            rotated_points_px[j, 1, 0], rotated_points_px[j, 1, 1],
                            rotated_points_px[j, 2, 0], rotated_points_px[j, 2, 1],
                            rotated_points_px[j, 3, 0], rotated_points_px[j, 3, 1],
                            class_boxes[j],
                            difficulty,
                            alpha[j], x1[j], y1[j], x2[j], y2[j], h3d[j], w3d[j], l3d[j], x3d[j], y3d[j], z3d[j], yaw_kitti[j]))
                lines_out.append(output_str)

        write_lines(path= os.path.join(output_label_folder, os.path.basename(label_file_path)), lines_with_return_character= lines_out)

        if (index+1) % 3000 == 0 or (index+1) == len(list_of_label_files):
            print("{:5d} images done.".format(index+1), flush= True)

#===================================================================================================
# KITTI-360 Dataset
#===================================================================================================
train_val_folder  = "data/kitti_360/train_val/label"
testing_folder    = "data/kitti_360/testing/label"

convert_to_dota(label_folder= train_val_folder)
convert_to_dota(label_folder= testing_folder)

#===================================================================================================
# KITTI Dataset
#===================================================================================================
# training_folder   = "data/KITTI/training/label_2"
# testing_folder    = "data/KITTI/testing/label_2"
#
# convert_to_dota(label_folder= training_folder)
# convert_to_dota(label_folder= testing_folder)