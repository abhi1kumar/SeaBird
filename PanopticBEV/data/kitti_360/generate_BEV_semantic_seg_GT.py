"""
    Sample Run:
    python data/kitti_360/generate_BEV_semantic_seg_GT.py

    Generates BEV semantic seg GT of the images.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import cv2
from PIL import Image
import glob

from panoptic_bev.helpers.file_io import read_image, write_image, write_lines, read_panoptic_dataset_binary
from panoptic_bev.helpers.more_util import get_semantic_GT

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import plot.plotting_params as params

kitti_360_folder = "data/kitti_360/KITTI-360"
bev_img_folder   = "data/kitti_360/kitti360_panopticbev/"
extension        = ".png"
bev_msk_folder   = os.path.join(bev_img_folder, "bev_msk/bev_ortho/")
metadata_path    = os.path.join(bev_img_folder, "metadata_ortho.bin")
sem_seg_folder   = os.path.join(bev_img_folder, "sem_msk")

if not os.path.exists(sem_seg_folder):
    os.makedirs(sem_seg_folder)

metadata = read_panoptic_dataset_binary(bin_path= metadata_path)

bev_msk_list   = sorted(glob.glob(bev_msk_folder + "/*.png"))
for i, bev_msk_path in enumerate(bev_msk_list):
    img_id          = os.path.basename(bev_msk_path).replace(extension, "")
    drive_name, img_name = img_id.split(";")

    # Get BEV semantic seg GT and save in panoptic BEV folder
    bev_msk         = Image.open(bev_msk_path)
    bev_msk         = bev_msk.rotate(90, Image.NEAREST, expand = 1)
    bev_panoptic    = np.array(bev_msk, dtype=np.int32, copy=False)
    bev_semantic, cat_img, calib_img = get_semantic_GT(img_id, bev_panoptic, metadata)

    bev_semantic_bgr = cv2.merge([bev_semantic, bev_semantic, bev_semantic])
    sem_seg_path     = os.path.join(sem_seg_folder, img_id + extension)
    write_image(sem_seg_path, bev_semantic_bgr)

    """
    # Sanity check to see if semantic segmentations are correct
    img_path        = "kitti_360/KITTI-360/data_2d_raw/"           + drive_name + "/image_00/data_rect/" + img_name + extension
    img              = imread(img_path)[:,:,::-1]
    temp             = imread(sem_seg_path)[:,:,0]
    temp             = np.array(temp)
    bev_semantic_new = np.copy(temp).astype(np.uint8)
    for i, cat_curr in enumerate(cat_img):
        if cat_curr == 255:
            cat_curr_new = 0
        else:
            cat_curr_new = cat_curr + 1
        bev_semantic_new[temp == cat_curr] = cat_curr_new

    fig = plt.figure(constrained_layout=True, dpi= params.DPI)
    gs = GridSpec(2, 2, figure=fig)

    fig.add_subplot(gs[0, :])
    plt.imshow(img)
    plt.title(img_id)
    plt.axis('off')

    fig.add_subplot(gs[-1, 0])
    plt.imshow(bev_semantic_new, 'viridis')
    plt.title('Sem')
    plt.axis('off')

    fig.add_subplot(gs[-1, 1])
    plt.imshow(bev_panoptic, cmap='viridis')
    plt.title('Panoptic')
    plt.axis('off')

    plt.show()
    plt.close()

    if i > 3:
        break
    """

    if (i+1) % 1000 == 0 or i == len(bev_msk_list)-1:
        print("{} images done".format(i+1))
