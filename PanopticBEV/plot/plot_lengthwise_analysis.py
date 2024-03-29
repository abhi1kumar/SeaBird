"""
    Sample Run:
    python plot/plot_lengthwise_analysis.py

    First run test/get_lengthwise_analysis.py with the appropriate folder and then run this.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
np.set_printoptions   (precision= 4, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

def finalize(data, reduce='sum'):
    if reduce == 'sum':
        return np.sum(data, axis=1)
    else:
        return np.mean(data, axis=1)

def plot_single_ap(all_ap_list, all_label_list, select, ylabel, save_path):
    base_ap_list    = [   all_ap_list[i] for i in select]
    base_label_list = [all_label_list[i] for i in select]

    monodle, gup_net, deviant, monodetr = base_ap_list
    monodle_mean = finalize(monodle, reduce= reduce)
    gup_net_mean = finalize(gup_net, reduce= reduce)
    deviant_mean = finalize(deviant, reduce= reduce)
    monodetr_mean = finalize(monodetr, reduce= reduce)
    i2m_seabird_mean = finalize(i2m_seabird, reduce= reduce)
    pbev_seabird_mean= finalize(pbev_seabird, reduce= reduce)

    xticks = (0.5*(length_bins[1:]+length_bins[:-1])).astype(np.uint8)
    yticks = np.arange(0, 21, 5)
    if reduce == 'sum':
        yticks = num_classes * yticks

    plt.figure(figsize= params.size, dpi=params.DPI)
    plt.plot(xticks, monodle_mean, lw= lw+1, c= colors[0], label= base_label_list[0])
    plt.plot(xticks, gup_net_mean, lw= lw, c= colors[1], label= base_label_list[1])
    plt.plot(xticks, deviant_mean, lw= lw, c= colors[2], label= base_label_list[2])
    plt.plot(xticks, monodetr_mean, lw= lw, c= colors[3], label= base_label_list[3])
    plt.plot(xticks, i2m_seabird_mean, lw= lw, c= colors[4], label='I2M+SeaBird')
    plt.plot(xticks, pbev_seabird_mean, lw= lw, c= colors[5], label='PBEV+SeaBird')
    plt.grid()
    plt.xticks(xticks, xticks)
    plt.yticks(yticks, yticks)
    plt.ylim(0, ymax)
    plt.xlabel("GT Object Length (m)")
    plt.ylabel(ylabel)
    plt.legend(fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
    savefig(plt, path= save_path)
    plt.close()

# =================================================================
# Main starts here
# =================================================================
length_max  = 20
ymax = 42
num_bins    = 4
num_classes = 2
reduce      = 'sum'
legend_fs   = params.legend_fs+15
lw          = params.lw + 3
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad+0.1
legend_vertical_label_spacing      = 0.0#params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing

length_bins    = np.arange(0, length_max + 0.01, step= length_max / num_bins)
colors = [params.color_set1_pink/255, params.color_seman_blue/255, params.color_set1_yellow/255]
colors = [params.color_seaborn_5, params.color_seaborn_4, params.color_seaborn_3, params.color_seaborn_15, params.color_seaborn_0, params.color_seaborn_1]

select = [2, 3, 4, 5]
all_label_list = ["GrooMeD-NMS", "MonoDLE", "GUP Net", "DEVIANT", "Cube R-CNN", "MonoDETR"]
# =================================================================
# IoU threshold = 0.50
# =================================================================
# gup_net = [[0, 8.72], [0, 42.11], [0.00, 0], [0.19, 0], [0.00, 0]]
# deviant = [[0, 7.37], [0, 41.31], [0.00, 0], [0.56, 0], [0.00, 0]]
# i2m_seabird = [[0, 2.22], [0, 40.27], [0.04, 0], [6.88, 0], [2.23, 0]]

groomed      = [[]]
monodle      = [[0, 37.17], [0, 0], [0.15, 0], [0.0, 0]]      # run_3
gup_net      = [[0, 37.77], [0, 10.91], [0.05, 0], [0.41, 0]] # gup_60
deviant      = [[0, 36.07], [0, 7.15], [0.00, 0], [0.46, 0]]  # dev_61
cubercnn     = [[0, 18.30], [0, 2.73], [2.23, 0], [2.29, 0]]  # kitti360_val
monodetr     = [[0, 36.29], [0, 0], [0.52, 0], [0.0, 0]]      # run_6
i2m_seabird  = [[0, 27.45], [0, 9.51], [5.56, 0], [5.66, 0]]  # run_645
pbev_seabird = [[0, 35.56], [0, 12.19], [6.67, 0], [8.70, 0]] # run_28

all_ap_list = [groomed, monodle, gup_net, deviant, cubercnn, monodetr]

ylabel    = r"$AP_{3D}50~(\%)~\uparrow$"
save_path = "images/lengthwise_analysis.png"
plot_single_ap(all_ap_list= all_ap_list, all_label_list= all_label_list, select= select, ylabel= ylabel, save_path= save_path)


# =================================================================
# IoU threshold = 0.25
# =================================================================
groomed      = [[]]
monodle      = [[0, 40.36], [0, 0], [1.50, 0], [0.0, 0]]         # run_3
gup_net      = [[0, 41.33], [0, 13.19], [0.05, 0], [1.28, 0]]    # gup_60
deviant      = [[0, 39.20], [0, 9.87], [0.00, 0], [1.10, 0]]     # dev_61
cubercnn     = [[0, 21.12], [0.02, 3.40], [7.67, 0], [7.36, 0]]  # kitti360_val
monodetr     = [[0, 39.29], [0, 0], [2.30, 0], [0.0, 0]]         # run_6
i2m_seabird  = [[0, 34.24], [0, 12.86], [15.66, 0], [17.37, 0]]  # run_645
pbev_seabird = [[0, 41.30], [0, 15.11], [11.04, 0], [16.95, 0]]  # run_28

all_ap_list = [groomed, monodle, gup_net, deviant, cubercnn, monodetr]

ylabel    = r"$AP_{3D}25~(\%)~\uparrow$"
save_path = "images/lengthwise_analysis_loose.png"
plot_single_ap(all_ap_list= all_ap_list, all_label_list= all_label_list, select= select, ylabel= ylabel, save_path= save_path)
