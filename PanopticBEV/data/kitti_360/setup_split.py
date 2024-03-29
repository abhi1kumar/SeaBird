"""
    Sample Run:
    python data/kitti_360/setup_split.py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from panoptic_bev.helpers.file_io import read_lines

BASE_PATH = "data/kitti_360"
IMAGE_EXT = ".png"

def make_symlink_or_copy(src_path, intended_path, MAKE_SYMLINK = True):
    if not os.path.exists(intended_path):
        if MAKE_SYMLINK:
            os.symlink(src_path, intended_path)
        else:
            command = "cp " + src_path + " " + intended_path
            os.system(command)

def link_original_data(inp_id_list_path, out_id_list_path, split= "trainval"):
    CWD = os.getcwd()
    # Read id lists
    inp_id_list = read_lines(inp_id_list_path)
    out_id_list = read_lines(out_id_list_path)

    assert len(inp_id_list) == len(out_id_list)

    split_folder     = "train_val" if "trainval" in split else "testing"
    out_image_folder = os.path.join(CWD, BASE_PATH, split_folder, "image")
    out_seman_folder = os.path.join(CWD, BASE_PATH, split_folder, "seman")
    out_panop_folder = os.path.join(CWD, BASE_PATH, split_folder, "panop")
    out_weght_folder = os.path.join(CWD, BASE_PATH, split_folder, "weght")
    out_front_folder = os.path.join(CWD, BASE_PATH, split_folder, "front")
    os.makedirs(out_image_folder, exist_ok= True)
    os.makedirs(out_seman_folder, exist_ok= True)
    os.makedirs(out_panop_folder, exist_ok= True)
    os.makedirs(out_front_folder, exist_ok= True)
    os.makedirs(out_weght_folder, exist_ok= True)

    cnt = 0
    for inp_id, out_id in zip(inp_id_list, out_id_list):
        out_image_path = os.path.join(out_image_folder, out_id + IMAGE_EXT)
        out_seman_path = os.path.join(out_seman_folder, out_id + IMAGE_EXT)
        out_panop_path = os.path.join(out_panop_folder, out_id + IMAGE_EXT)
        out_weght_path = os.path.join(out_weght_folder, out_id + IMAGE_EXT)
        out_front_path = os.path.join(out_front_folder, out_id + IMAGE_EXT)

        drive, diid    = inp_id.split(";")
        if "testing" in split:
            # Map to one of the trainval ids
            inp_id = "2013_05_28_drive_0004_sync;0000009808"

        inp_image_path = os.path.join(CWD, BASE_PATH, "KITTI-360/data_2d_raw", drive, "image_00/data_rect", diid + IMAGE_EXT)
        inp_seman_path = os.path.join(CWD, BASE_PATH, "kitti360_panopticbev/sem_msk"                , inp_id + IMAGE_EXT)
        inp_panop_path = os.path.join(CWD, BASE_PATH, "kitti360_panopticbev/bev_msk/bev_ortho"      , inp_id + IMAGE_EXT)
        inp_weght_path = os.path.join(CWD, BASE_PATH, "kitti360_panopticbev/class_weights"          , inp_id + IMAGE_EXT)
        inp_front_path = os.path.join(CWD, BASE_PATH, "kitti360_panopticbev/front_msk_trainid/front", inp_id + IMAGE_EXT)

        make_symlink_or_copy(src_path= inp_image_path, intended_path= out_image_path)
        make_symlink_or_copy(src_path= inp_seman_path, intended_path= out_seman_path)
        make_symlink_or_copy(src_path= inp_panop_path, intended_path= out_panop_path)
        make_symlink_or_copy(src_path= inp_weght_path, intended_path= out_weght_path)
        make_symlink_or_copy(src_path= inp_front_path, intended_path= out_front_path)

        cnt += 1
        if cnt % 5000 == 0 or out_id == out_id_list[-1]:
            print("{} images done...".format(cnt))

#===================================================================================================
# Main starts here
#===================================================================================================
# Link train
print('=============== Linking trainval =======================')
inp_id_list_path = "data/kitti_360/ImageSets/org_trainval_det_clean.txt"
out_id_list_path = "data/kitti_360/ImageSets/trainval_det.txt"
link_original_data(inp_id_list_path, out_id_list_path, split= "trainval")

# Link test
print('=============== Linking test =======================')
inp_id_list_path = "data/kitti_360/ImageSets/org_test_det_samp.txt"
out_id_list_path = "data/kitti_360/ImageSets/test_det.txt"
link_original_data(inp_id_list_path, out_id_list_path, split= "testing")



