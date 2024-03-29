"""
    Sample Run:
    python plot/plot_qualitative_output.py --folder output/run_1/results_test/data   --dataset nuscenes
    python plot/plot_qualitative_output.py --folder data/kitti_360/validation/label/ --dataset kitti_360 --show_gt_in_image
    python plot/plot_qualitative_output.py --folder output/pbev_seabird_kitti360_val/result_19_val_det_samp/data --folder2 /home/abhinav/project/MonoDETR_kitti_360/output/run_6/result_val_det_samp/data --dataset kitti_360 --compression 20
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import ImageFont, Image

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

import cv2
import imutils
import random
import copy

from panoptic_bev.helpers.file_io import read_image, write_image, read_numpy
from panoptic_bev.helpers.kitti_utils import get_objects_from_label, get_calib_from_file
from panoptic_bev.helpers.more_util import (convertRot2Alpha, imhstack, project_3d, draw_3d_box,
                                            draw_bev, draw_tick_marks, create_colorbar, draw_2d_box,
                                            draw_filled_rectangle, draw_text, draw_border)

import glob
import argparse

# ==================================================================================================
# COLORS
# https://sashamaps.net/docs/resources/20-colors/
# Some taken from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py#L10
# ==================================================================================================
class_color_map = {'car': (255, 204, 51),  #(252, 232, 168), #(51, 153, 255),
                   'cyclist': (255, 130, 48),  # Orange
                   'bicycle': (255, 130, 48),  # Orange
                   'pedestrian': (138, 43, 226),  # Violet
                   'bus': (0, 0, 0),  # Black
                   'construction_vehicle': (0, 130, 200),  # Blue
                   'motorcycle': (220, 190, 255),  # Lavender
                   'trailer': (170, 255, 195),  # Mint
                   'truck': (128, 128, 99),  # Olive
                   'traffic_cone': (255, 225, 25),  # Yellow
                   'barrier': (128, 128, 128),  # Grey
                   'building': (51, 153, 255), # (115, 31, 130), #(255, 204, 51),  # Golden Yellow
                   }

bev_c2       = (255, 255, 255)#(0, 175, 250)
bev_c1       = (204, 204, 204)#(0, 250, 250)
c_gts        = (10, 175, 10)
c            = (255,51,153)#(255,48,51)#(255,0,0)#(114,211,254) #(252,221,152 # RGB

color_gt     = (52 ,168, 83)#(0, 255 , 0)
color_pred_2 = (191, 56, 117)#(59, 221, 255)#(51,153,255)#(94,45,255)#(255, 128, 0)
use_classwise_color = True

def plot_boxes_on_image_and_in_bev(predictions_img, plot_color, box_class_list= ["car", "cyclist", "pedestrian"], use_classwise_color= False, show_3d= True, show_bev= True, show_2d_box= False, thickness= 4):
    if predictions_img is not None and predictions_img.size > 0:
        # Add dimension if there is a single point
        if predictions_img.ndim == 1:
            predictions_img = predictions_img[np.newaxis, :]

        N = predictions_img.shape[0]
        #   0   1    2     3   4   5  6    7    8    9    10   11   12   13    14      15
        # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, score/num_lidar_points  )
        # Add projected 3d center information to predictions_img
        cls = predictions_img[:, 0]
        x1  = predictions_img[:, 4]
        y1  = predictions_img[:, 5]
        x2  = predictions_img[:, 6]
        y2  = predictions_img[:, 7]
        h3d = predictions_img[:, 8]
        w3d = predictions_img[:, 9]
        l3d = predictions_img[:, 10]
        x3d = predictions_img[:, 11]
        y3d = predictions_img[:, 12] - h3d/2
        z3d = predictions_img[:, 13]
        ry3d = predictions_img[:,14]

        if predictions_img.shape[1] > 15:
            score = predictions_img[:, 15]
        else:
            score = np.ones((predictions_img.shape[0],))

        for j in range(N):
            box_class = cls[j].lower()
            if box_class == "dontcare":
                continue
            # For predictions do not plot close
            if use_classwise_color and z3d[j] <= 4:
                continue
            # if score[j] < 0.2:
            #     continue
            if dataset == "nuscenes" or box_class in box_class_list:
                if use_classwise_color:
                    box_plot_color = class_color_map[box_class]
                else:
                    box_plot_color = plot_color

                # if box_class == "car" and score[j] < 100:
                #     continue
                # if box_class != "car" and score[j] < 50:
                #     continue

                box_plot_color = box_plot_color[::-1]
                if show_3d:
                    verts_cur, corners_3d_cur = project_3d(p2, x3d[j], y3d[j], z3d[j], w3d[j], h3d[j], l3d[j], ry3d[j], return_3d=True)
                    if show_2d_box:
                        # draw_2d_box(img, verts_cur, color= box_plot_color, thickness= thickness)
                        draw_2d_box(img, [x1[j], y1[j], x2[j], y2[j]], color= box_plot_color, thickness= thickness, verts_as_corners= False)
                    else:
                        draw_3d_box(img, verts_cur, color= box_plot_color, thickness= thickness)
                    # cv2.putText(img, str(int(10*score[j])), (int(x1[j]), int(y1[j])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
                if show_bev:
                    draw_bev(canvas_bev, z3d[j], l3d[j], w3d[j], x3d[j], ry3d[j], color= box_plot_color, scale= bev_scale, thickness= thickness+5, text= None)#str(int(score[j])))

#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='plot qualitative outputs DEVIANT')
parser.add_argument('--dataset', type=str, default = "kitti", help='one of kitti,nusc_kitti,nuscenes,waymo,kitti_360')
parser.add_argument('--folder' , type=str, default = "output/deviant/results_test/data", help='evaluate model on validation set')
parser.add_argument('--folder2', type=str, default=  "output/prevolution_kitti_full/results_nusc_kitti_val/data")
parser.add_argument('--gt_folder',type=str, default = "", help='gt_folder')
parser.add_argument('--compression', type=int, default= 100)
parser.add_argument('--show_gt_in_image', action= 'store_true', default= False, help='show ground truth 3D boxes in image for sanity check')
parser.add_argument('--show_2d', action= 'store_true', default= False, help='show projected 2D boxes')
args = parser.parse_args()


dataset  = args.dataset
folder   = args.folder
folder2  = args.folder2
compression_ratio = args.compression

show_ground_truth = True
show_baseline     = True
show_gt_in_image  = args.show_gt_in_image
show_2d_box       = args.show_2d
video_demo_flag   = "video_demo" in args.folder
num_files_to_plot = 200
show_seg          = True

seed = 0
np.random.seed(seed)
random.seed(seed)
bev_scale    = 20   # Pixels per meter
bev_max_w    = 20   # Max along positive X direction. # This corresponds to the camera-view of (-max, max)
bev_w        = 2 * bev_max_w * bev_scale

# ==================================================================================================
# Dataset Specific Settings
# ==================================================================================================
if dataset == "kitti":
    box_class_list= ["car", "cyclist", "pedestrian"]
    lidar_points_in_gt = False

    if args.gt_folder != "":
        gt_folder = args.gt_folder
    elif video_demo_flag:
        gt_folder = "data/KITTI/video_demo"
    else:
        gt_folder = "data/KITTI/training"
    img_folder   = os.path.join(gt_folder, "image_2")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label_2")

    zfill_number = 6
    if video_demo_flag:
        print("Running with video_demo settings...")
        zfill_number = 10

    bev_max_z= 50
    ticks = [50, 40, 30, 20, 10, 0]

elif dataset == "kitti_360":
    cls2id = {'building': 2, 'car': 9}
    id2cls = {v: k for k, v in cls2id.items()}
    box_class_list= ["car", "cyclist", "pedestrian", "building"]
    lidar_points_in_gt = False

    gt_folder = "data/kitti_360/train_val"
    img_folder   = os.path.join(gt_folder, "image")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label")

    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]


elif dataset == "nusc_kitti":
    box_class_list= ["car", "bicycle", "pedestrian", "barrier", "bus", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck"]
    lidar_points_in_gt = True

    gt_folder = "data/nusc_kitti/validation"
    img_folder   = os.path.join(gt_folder, "image")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label")

    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

elif dataset == "waymo":
    box_class_list= ["car", "cyclist", "pedestrian"]
    lidar_points_in_gt = True

    gt_folder = "data/waymo/validation"
    img_folder   = os.path.join(gt_folder, "image")
    cal_folder   = os.path.join(gt_folder, "calib")
    lab_folder   = os.path.join(gt_folder, "label")

    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

elif dataset == "nuscenes":
    cls2id = {'barrier': 0, 'bicycle': 1, 'bus': 2, 'car': 3, 'construction_vehicle': 4, 'motorcycle': 5, 'pedestrian': 6, 'traffic_cone': 7, 'trailer': 8, 'truck': 9}
    id2cls = {v: k for k, v in cls2id.items()}
    gt_folder = None
    # load_pkl
    data_cache_path = os.path.join(os.getcwd(), 'nuscenes_val.pkl')
    data_cache      = pickle_read(file_path= data_cache_path)

    bev_w = int(1.6* bev_w)
    bev_max_z= 80
    ticks = [80, 70, 60, 50, 40, 30, 20, 10, 0]

print(box_class_list)

# ==================================================================================================
# Output
# ==================================================================================================
img_save_folder   = "images/qualitative/" + folder.split("/")[1] + "_" + dataset + "_" + folder.split("/")[2]  #"waymo_with_gt_segment-14739149465358076158_4740_000_4760_000_with_camera_labels_new"
if video_demo_flag:
    img_save_folder += "/" + gt_folder.split("/")[-1]
os.makedirs(img_save_folder,exist_ok= True)

files_list   = sorted(glob.glob(folder + "/*.txt"))
num_files    = len(files_list)

print("Choosing {} files out of {} files...".format(num_files_to_plot, num_files))

if video_demo_flag:
    file_index = np.arange(num_files)
else:
    file_index = np.sort(np.random.choice(range(num_files), num_files_to_plot, replace=False))


for i in range(file_index.shape[0]):
    curr_index = file_index[i]
    pred_file  = files_list[curr_index]
    basename   = os.path.basename(pred_file).split(".")[0]
    # basename   = str(file_index[i]).zfill(6)
    # pred_file  = os.path.join(args.folder, basename+".txt")

    # basename   = str(file_index[i]).zfill(6)
    # pred_file  = os.path.join(args.folder, basename + ".txt")

    int_id = int(basename)
    # if (int_id >= 54306 and int_id <= 54569) or (int_id >= 55377 and int_id <= 55500) \
    #     or (int_id >= 55705 and int_id <= 55808) or (int_id >= 66343 and int_id <= 66398):
    #     pass
    # else:
    #     continue

    if dataset == "nuscenes":
        #match_by_file_name
        matched_index = [t for t in range(len(data_cache)) if (os.path.basename(data_cache[t]['file_name']).split(".")[0] ==  basename)]
        matched_index = matched_index[0]
        img_file   = data_cache[matched_index]['file_name']

        p2_temp    = np.array(data_cache[matched_index]['intrinsics']).reshape(3,3)
        p2         = np.eye(4)
        p2[:3, :3] = p2_temp

        annotations     = data_cache[matched_index]['annotations']
        num_annotations = len(annotations)
        lines = []
        for i, anno_curr in enumerate(annotations):
            category_name = id2cls[anno_curr['category_id']]
            cx, cy, cz    = anno_curr['box3d']['center']
            w, l, h       = anno_curr['box3d']['wlh']
            ry3d          = anno_curr['box3d']['yaw']
            alpha         = convertRot2Alpha(ry3d= ry3d, z3d= cz, x3d= cx)
            x1, y1, x2, y2= anno_curr['box2d']
            # Label format as KITTI
            #   0   1    2     3   4   5  6    7    8    9    10   11   12         13    14   15
            # (cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d + h/2, z3d, ry3d, score,
            line = [category_name, 0, 0, alpha, x1, y1, x2, y2, h, w, l, cx, cy + h/2.0, cz, ry3d]
            lines.append(line)
        gt_img = pd.DataFrame(lines).values

    else:
        # if dataset == "kitti":
            # if not video_demo_flag and int(basename) not in   [35, 5086, 4485, 3207, 1868, 1101, 3135]:
            #     continue

        cal_file   = os.path.join(cal_folder, basename + ".txt")
        img_file   = os.path.join(img_folder, basename + ".png")
        label_file = os.path.join(lab_folder, basename + ".txt")
        p2         = get_calib_from_file(cal_file)['P2']
        gt_img     = read_csv(label_file, ignore_warnings= True, use_pandas= True)

        if gt_img is not None:
            gt_other      = gt_img[:, 1:].astype(float)
            #       0  1   2      3   4   5   6    7    8    9   10   11   12    13     14
            # cls, -1, -1, alpha, x1, y1, x2, y2, h3d, w3d, l3d, x3d, y3d, z3d, ry3d, lidar
            if lidar_points_in_gt:
                gt_bad_index  = np.logical_or(np.logical_or(np.logical_or(gt_other[:, 14] == 0, gt_other[:, 3]-gt_other[:, 5] >=0), gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )
            else:
                gt_bad_index  = np.logical_or(np.logical_or(gt_other[:, 3]-gt_other[:, 5] >=0, gt_other[:, 4]-gt_other[:, 6] >=0), gt_other[:, 12] <=0 )

            gt_good_index = np.logical_not(gt_bad_index)
            gt_img        = gt_img[gt_good_index]

    img        = read_image(img_file)

    canvas_bev = create_colorbar(bev_max_z * bev_scale, bev_w, color_lo=bev_c1, color_hi=bev_c2)

    if show_gt_in_image:
        plot_boxes_on_image_and_in_bev(gt_img, plot_color= color_gt, box_class_list= box_class_list, use_classwise_color= use_classwise_color, show_3d= True, show_2d_box= show_2d_box)
    else:
        if show_ground_truth:
            plot_boxes_on_image_and_in_bev(gt_img, plot_color= color_gt, box_class_list= box_class_list, show_3d= False, show_2d_box= show_2d_box)

    if show_baseline:
        predictions_file_2     = os.path.join(folder2, basename + ".txt")
        predictions_img_2      = read_csv(predictions_file_2, ignore_warnings= True, use_pandas= True)
        plot_boxes_on_image_and_in_bev(predictions_img_2, plot_color= color_pred_2, box_class_list= box_class_list, show_3d= False, show_2d_box= show_2d_box, thickness= 8)

    if not show_gt_in_image:
        predictions_img  = read_csv(pred_file, ignore_warnings= True, use_pandas= True)
        if predictions_img is not None and gt_img is not None:
            plot_boxes_on_image_and_in_bev(predictions_img, plot_color = c, box_class_list= box_class_list, use_classwise_color= use_classwise_color, show_2d_box= show_2d_box, thickness= 6)
            print("Predictions = {} GT= {}".format(predictions_img.shape[0], gt_img.shape[0]))

    canvas_bev = cv2.flip(canvas_bev, 0)
    draw_text(canvas_bev, 'BEV Det', (150, 80), lineType= cv2.LINE_8, font= cv2.FONT_HERSHEY_DUPLEX, scale=3.0, bg_color=None)
    # draw tick marks
    draw_tick_marks(canvas_bev, ticks)
    # add border on right
    canvas_bev = draw_border(image= canvas_bev, thickness= 4, style= 'right')
    # concat frontal and bev
    im_concat = imhstack(img, canvas_bev)

    seg_folder     = os.path.join("/".join(os.path.dirname(args.folder).split("/")[:2]), "seg")
    if os.path.exists(seg_folder):
        sem_seg_path   = os.path.join(seg_folder, basename + ".npy")
        seg            = np.rot90(read_numpy(sem_seg_path))
    else:
        seg            = np.zeros((704, 768)).astype(np.uint8)
    if show_seg:
        print("Adding sematic segmentation outputs as well...")
        seg_map        = 255*np.ones((704, 768, 3)).astype(np.uint8)
        for sem_index in [2, 9]:
            sem_name       = id2cls[sem_index]
            seg_color      = class_color_map[sem_name]
            sem_index      = np.where(seg == sem_index)
            seg_map[sem_index] = seg_color
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2BGR)

        seg_bev_h = 1025
        seg_bev_w = int(768/704*seg_bev_h)
        det_bev_h = canvas_bev.shape[0] # 1600
        det_bev_w = canvas_bev.shape[1] # 800
        seg_map     = cv2.resize(seg_map, (seg_bev_w, seg_bev_h), interpolation = cv2.INTER_AREA) # W x H x 3
        if seg_bev_w > det_bev_w:
            diff    = (seg_bev_w - det_bev_w)//2
            seg_map = seg_map[:, diff:-diff]
        blank_image = 255 * np.ones((det_bev_h, det_bev_w, 3), np.uint8) #H x W x 3 = 1600 x 800 x 3
        blank_image[det_bev_h-seg_bev_h:] = seg_map
        # create canvas and blend them
        sem_seg_canvas = cv2.flip(create_colorbar(bev_max_z * bev_scale, bev_w, color_lo=bev_c1, color_hi=bev_c2), 0)
        blank_image_gray = np.mean(blank_image, axis= 2).astype(np.uint8)
        index = blank_image_gray == 255
        blank_image[index] = sem_seg_canvas[index]
        # draw tick marks
        draw_tick_marks(blank_image, ticks)
        draw_text(blank_image, 'BEV Seg', (150, 80), lineType= cv2.LINE_8, font= cv2.FONT_HERSHEY_DUPLEX, scale=3.0, bg_color=None)
        # add border on left
        blank_image = draw_border(image= blank_image, thickness= 4, style= 'left')
        # concat frontal, bev with bev segmentation
        im_concat = imhstack(im_concat, blank_image)

        # Add legend
        blend = 0.0
        border = 10
        leg_h = 90
        leg_w = 110
        xs1 = 5520
        ys1 = 50
        xs2 = xs1
        ys2 = ys1 + leg_h
        xs3 = xs1
        ys3 = ys2 + leg_h
        im_concat = draw_filled_rectangle(im_concat, xs1-20, xs1+int(4*leg_w), ys1-20, ys1+3*leg_h, bg_color= (255, 255, 255), blend= blend, border= 5)
        im_concat = draw_filled_rectangle(im_concat, xs1, xs1+leg_w, ys1, ys1+50, bg_color= (255, 255, 255), border_color= class_color_map['building'][::-1], blend= blend, border= border)
        im_concat = draw_filled_rectangle(im_concat, xs2, xs2+leg_w, ys2, ys2+50, bg_color= (255, 255, 255), border_color= class_color_map['car'][::-1], blend= blend, border= border)
        im_concat = draw_filled_rectangle(im_concat, xs3, xs3+leg_w, ys3, ys3+50, bg_color= (255, 255, 255), border_color= color_gt[::-1], blend= blend, border= border)
        draw_text(im_concat, 'Building', (xs1+leg_w + 10, ys1+48), lineType= 2, font= cv2.FONT_HERSHEY_DUPLEX, scale=2.5, bg_color=None)
        draw_text(im_concat, 'Car' , (xs2+leg_w + 10, ys2+48), lineType= 2, font= cv2.FONT_HERSHEY_DUPLEX, scale=2.5, bg_color=None)
        draw_text(im_concat, 'GT'  , (xs3+leg_w + 10, ys3+48), lineType= 2, font= cv2.FONT_HERSHEY_DUPLEX, scale=2.5, bg_color=None)

    save_path = os.path.join(img_save_folder, basename + ".png")
    print("Saving to {} with compression ratio {}".format(save_path, compression_ratio))
    write_image(save_path, im_concat)

    if compression_ratio != 100:
        # Save smaller versions of image
        command = "convert -resize " + str(compression_ratio) + "% " + save_path + " " + save_path
        os.system(command)

    pass

# Video conversion
# r= framerate
# ffmpeg -safe 0 -r 5 -f concat -i file -q:v 1 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p images/out.mp4
# /snap/bin/ffmpeg -safe 0 -r 25 -f concat -i file -q:v 1 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -vcodec libx265 images/equivariance_error_demo.mp4
