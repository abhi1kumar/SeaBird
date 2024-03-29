

"""
    Stores the plotting settings to be used for plots.
    Import this module so that Tkinter or Agg can be checked

    Version 1 2020-05-25 Abhinav Kumar
"""
import matplotlib

# Check if Tkinter is there otherwise Agg
import imp
try:
    imp.find_module('_Tkinter')
    pass
except ImportError:
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

DPI        = 200
ms         = 12
lw         = 5
alpha      = 0.9
size       = (10, 6)

# Remember to call update whenever you use a different fontsize
fs         = 28
matplotlib.rcParams.update({'font.size': fs})

# (length, space)
# example
# plt.plot(x, y, '--', dashes= params.dashes)
dashes      = (3, 0.6)

# legend properties
# example
# plt.legend(handles= legend_handles, loc= "upper right", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
legend_fs  = 16
legend_border_axes_plot_border_pad = 0.05
legend_border_pad                  = 0.1
legend_vertical_label_spacing      = 0.1
legend_marker_text_spacing         = 0.2

IMAGE_DIR  = "images"
dodge_blue =  np.array([0.12, 0.56, 1.0]) #np.array([30, 144, 255])/255.
color1     = (1,0.45,0.45)
color2     = "dodgerblue"

# basic ones
color_red    = np.array([255, 0, 0.0])
color_blue   = np.array([0, 0, 255.0])
color_yellow = np.array([255, 255, 0.0])

#triadic
color_set1_cyan   = np.array([59, 221, 255.0])
color_set1_pink   = np.array([255, 59, 141.0])
color_set1_yellow = np.array([255, 216, 59.0])

color_set1_cyan_light   = np.array([177, 225, 255.0])
color_set1_pink_light   = np.array([255, 177, 200.0])
color_set1_yellow_light = np.array([255, 243, 77.0])

num_bins = 50

color_set2_cyan        = np.array([0, 59, 157.0])
color_set2_cyan_light  = np.array([145,236, 255.0])
color_set2_yellow      = np.array([255, 180, 0.0])
color_set2_yellow_light= np.array([255, 255, 0.0])
color_set2_pink        = np.array([254, 0.0, 0.0])
color_set2_pink_light  = np.array([255, 213, 225.0])

display_frequency = 500
min_iou2d_overlap = 0.5
frac_to_keep      = 0.7

semantic_color_map = {'0': np.array([0, 0, 0]), '1': np.array([255, 0, 0])}
color_seman_blue     = np.array([100, 150, 245.0])
color_seman_building = np.array([47, 79, 79.0])

import seaborn as sns
colors_temp = sns.color_palette("magma", 20)
color_seaborn_0 = colors_temp[3]
color_seaborn_1 = colors_temp[6]
color_seaborn_15= colors_temp[10]
color_seaborn_2 = colors_temp[13]
color_seaborn_3 = colors_temp[15]
color_seaborn_4 = colors_temp[17]
color_seaborn_5 = colors_temp[19]

# Semantic Segmentation Colors
# https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
# categories= {0: 'invalid',
# 1: 'occlusion',
# 10: 'car',
# 11: 'rider',
# 13: 'truck',
# 30: 'person',
# 40: 'road',
# 48: 'sidewalk',
# 50: 'building',
# 51: 'wall',
# 70: 'vegetation',
# 72: 'terrain',
# }

# RGB
# semantic_color_map = {0: [0, 0, 0], 1: [255, 0, 0]}
# 10: [100, 150, 245],
# 11: [100, 230, 245],
# 13: [100, 80, 250],
# 30: [255, 30, 30],
# 40: [255, 0, 255],
# 48: [75, 0, 75],
# 50: [255, 200, 0],
# 51: [255, 120, 50],
# 70: [0, 175, 0],
# 72: [150, 240, 80],
# }
