

import os, sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from plot.common_operations import *
import plot.plotting_params as params

legend_fs  = params.legend_fs + 10
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing
matplotlib.rcParams.update({'font.size': 26})

# KITTI-360 performance AP_lrg (Buildings), mAP
gupnet_kitti_360      = np.array([0.54, 22.83])
deviant_kitti_360     = np.array([0.53, 22.39])
monodle_kitti_360     = np.array([0.94, 22.88])
monodetr_kitti_360    = np.array([0.81, 22.02])
i2m_seabird_kitti_360 = np.array([8.71, 25.95])
pan_seabird_kitti_360 = np.array([13.22, 27.84])
# nuScenes performance AP_lrg, mAP, NDS
beverse_nusc          = np.array([20.9, 35.2, 49.5])
beverse_seabird_nusc  = np.array([24.6, 38.2, 51.3])
hop_nusc              = np.array([36.5, 49.6, 58.3])
hop_seabird_nusc      = np.array([40.3, 52.7, 60.2])

colors = [params.color_set1_pink/255., "dodgerblue", np.array([59, 255, 173.0])/255., params.color_set1_yellow/255.]#params.color_set2_yellow/255.]
colors = [params.color_seaborn_5, params.color_seaborn_4, params.color_seaborn_15, params.color_seaborn_1]
edgecolor = 'k'
# ==================================================================================================
# KITTI-360 Plots
# ==================================================================================================
width = 0.25
ymax = 30.5
x = np.arange(2)*1.25#*width

plt.figure(figsize=params.size, dpi=params.DPI)
labels = ["MonoDLE", "GUP Net", "DEVIANT", "MonoDETR", "I2M+SeaBird", "PBEV+SeaBird"]
plt.bar(x-width*1.5, deviant_kitti_360 , width, color=colors[0], label=labels[2], edgecolor= edgecolor, zorder=3)
plt.bar(x-width*0.5, monodetr_kitti_360 , width, color=colors[1], label=labels[3], edgecolor= edgecolor, zorder=3)
plt.bar(x+width*0.5, i2m_seabird_kitti_360, width, color=colors[2], label=labels[4], edgecolor= edgecolor, zorder=3)
plt.bar(x+width*1.5, pan_seabird_kitti_360, width, color=colors[3], label=labels[5], edgecolor= edgecolor, zorder=3)
plt.grid(zorder=0)
plt.ylim(0, ymax)
plt.ylabel("AP" + r"$_{3D}50~(\%)~\uparrow$")
ax= plt.gca()
ax.set_xticks(x)
ax.set_xticklabels([r"AP$_{Lrg}$", "mAP"])
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f')
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
plt.legend(loc= "upper left", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, "images/teaser_kitti_360.png")
#plt.show()
plt.close()

# ==================================================================================================
# nuScenes Plots
# ==================================================================================================
width = 0.25
ymin = 19
ymax = 64
x = np.arange(3)*1.25#*width
plt.figure(figsize=(15,6), dpi=params.DPI)
labels = ["BEVerse-S", "BEVerse-S+SeaBird", "HoP", "HoP+SeaBird"]
plt.bar(x-width*1.5, beverse_nusc        , width, color=colors[0], label=labels[0], edgecolor= edgecolor, zorder=3)
plt.bar(x-width*0.5, beverse_seabird_nusc, width, color=colors[2], label=labels[1], edgecolor= edgecolor, zorder=3)
plt.bar(x+width*0.5, hop_nusc            , width, color=colors[1], label=labels[2], edgecolor= edgecolor, zorder=3)
plt.bar(x+width*1.5, hop_seabird_nusc    , width, color=colors[3], label=labels[3], edgecolor= edgecolor, zorder=3)
plt.grid(zorder=0)
plt.ylim(ymin, ymax)
plt.ylabel("Metrics" + r"$~(\%)~\uparrow$")
ax= plt.gca()
ax.set_xticks(x)
ax.set_xticklabels([r"AP$_{Lrg}$", "mAP", "NDS"])
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f')
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
plt.legend(loc= "upper left", fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, "images/teaser_nuscenes.png")
# plt.show()
