"""
    Sample Run:
    python plot/plot_category_wise_stats.py

    Plots category wise stats of kitti360 and nuscenes datasets.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn

np.set_printoptions   (precision= 4, suppress= True)
torch.set_printoptions(precision= 4, sci_mode= False)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib


labels = ['KITTI-360', 'nuScenes']
legend_fs  = params.legend_fs +10
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing

# KITTI-360
kitti_360_cls_cnt = [114535, 188598, 0]# 8312]
# nuscenes
nuscenes_cls_cnt = [4.2e+4, 4.4e+5, 3.9e+5]

xticklabels      = ["Large", "Car", "Small"]
import seaborn as sns
colors_sns = sns.color_palette("magma", 10)
colors = [colors_sns[9], "dodgerblue", colors_sns[6], colors_sns[3]]
colors = [params.color_seaborn_5, "dodgerblue", params.color_seaborn_15]
edgecolor = 'k'
width = 0.7
ymax = 4.5e+5

x = np.arange(3)
y = (1 + np.arange(4))*1e+5
yticklabels = ['1', '2', '3', '4']

plt.figure(figsize=(12, 6), dpi=params.DPI)
plt.subplot(121)
barlist=plt.bar(x, kitti_360_cls_cnt, width, label=labels[0], edgecolor= edgecolor, zorder=3)
barlist[0].set_color(colors[0])
barlist[1].set_color(colors[1])
barlist[2].set_color(colors[2])
plt.grid(zorder=0)
plt.ylim(0, ymax)
plt.ylabel("Counts " + r'$(\times 10^5)$')
ax= plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
ax.set_yticks(y)
ax.set_yticklabels(yticklabels)
# for c in ax.containers:
#     ax.bar_label(float(c)/1e+5, fmt='%.1f')
plt.title('KITTI-360')
# plt.legend(loc= "upper right", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
plt.subplot(122)
barlist=plt.bar(x, nuscenes_cls_cnt, width, label=labels[1], edgecolor= edgecolor, zorder=3)
barlist[0].set_color(colors[0])
barlist[1].set_color(colors[1])
barlist[2].set_color(colors[2])
plt.grid(zorder=0)
plt.ylim(0, ymax)
ax= plt.gca()
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
ax.set_yticks(y)
ax.set_yticklabels(yticklabels)
# for c in ax.containers:
#     ax.bar_label(c, fmt='%.1f')
plt.title('nuScenes')
# plt.legend(loc= "upper right", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)


savefig(plt, "images/category_wise_stats.png", pad_inches= 0.2)
plt.close()
