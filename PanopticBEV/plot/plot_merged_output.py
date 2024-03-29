"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import argparse

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib
import cv2

from panoptic_bev.helpers.file_io import read_image, write_image
from panoptic_bev.helpers.more_util import draw_text
from PIL import ImageFont, ImageDraw, Image


#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='implementation of DEVIANT')
parser.add_argument('--folder'  , type= str, default= "images/qualitative/seabird_kitti360_demo_renamed", help='method')
parser.add_argument('--baseline', type= str, default= "images/qualitative/monodetr_kitti360_demo_renamed", help='baseline')
args   = parser.parse_args()

method_folder = args.folder
baseln_folder = args.baseline
output_folder = "images/qualitative/merged_kitti360_demo"
os.makedirs(output_folder, exist_ok= True)

method_files_list = sorted(glob.glob(method_folder + "/*.png"))
method_num_files  = len(method_files_list)
print("Found {} files in {}".format(method_num_files, method_folder))

baseln_files_list = sorted(glob.glob(baseln_folder + "/*.png"))
baseln_num_files  = len(baseln_files_list)
print("Found {} files in {}".format(baseln_num_files, baseln_folder))

assert method_num_files == baseln_num_files

text_x = 100
text_y = 250
lw      = 2
pad     = 35
border  = 8
scale   = 8
lineType= cv2.LINE_8
font    = cv2.FONT_HERSHEY_DUPLEX
blend   = 0.0

c1 = (np.array(params.color_seaborn_15) * 255).astype(np.uint8)
c2 = (np.array(params.color_seaborn_1) * 255).astype(np.uint8)
cbg = (np.array(params.color_seaborn_5) * 255).astype(np.uint8)


for i, (baseln_file, method_file) in enumerate(zip(baseln_files_list, method_files_list)):
    baseln_img = read_image(baseln_file, rgb= True)
    method_img = read_image(method_file, rgb= True)

    h1, w1, c = baseln_img.shape
    h2, w2, c = method_img.shape

    max_w = np.max([w1, w2])
    merged_image = 224* np.ones((h1+h2, max_w, c)).astype(np.uint8)
    merged_image[:h1, :w1] = baseln_img
    merged_image[h1:, :w2] = method_img
    savepath = os.path.join(output_folder, os.path.basename(baseln_file))

    # fig = plt.figure(figsize= (7592, 3200))#, dpi= params.DPI)
    # plt.imshow(merged_image)
    # plt.axis('off')
    # props = dict(boxstyle= 'round', facecolor= params.color_seaborn_5, alpha=1.0)
    # plt.text(text_x,    text_y, 'MonoDETR'    , bbox= props, color= params.color_seaborn_15)#dict(fill= False, facecolor= params.color_seaborn_15, alpha=0.5, linewidth= lw))
    # plt.text(text_x, h1+text_y, 'PBEV+SeaBird', bbox= props, color= params.color_seaborn_1)#dict(fill= False, facecolor= params.color_seaborn_1 , alpha=0.5, linewidth= lw))
    # savefig(plt, savepath)
    # plt.show()

    draw_text(merged_image, 'MonoDETR'    , (text_x,    text_y), color= (int(c1[0]), int(c1[1]), int(c1[2])), lineType= lineType, font= font, scale= scale, bg_color= cbg, pad= pad, blend= blend, border= border)
    draw_text(merged_image, 'PBEV+SeaBird', (text_x, h1+text_y), color= (int(c2[0]), int(c2[1]), int(c2[2])), lineType= lineType, font= font, scale= scale, bg_color= cbg, pad= pad, blend= blend, border= border)

    merged_image = merged_image[:, :, ::-1]
    write_image(savepath, merged_image)
    # break