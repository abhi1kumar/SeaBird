"""
    Sample Run:
    python plot/plot_convergence_analysis.py

    Plots log of variance of loss wrt noise at different noise levels.
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from scipy.special import erf

np.set_printoptions   (precision= 4, suppress= True)

from plot.common_operations import *
import plot.plotting_params as params
import matplotlib

def gradient(input, length= 4, name= "L2"):
    output = np.zeros(input.shape)
    if name == "l1":
        output = np.sign(input)
    elif name == "l2":
        output = input
    elif name == "smoothL1":
        index = input <= 1
        output[index] = input[index]
        index = input > 1
        output[index] = np.sign(input[index])
    elif name == "log_depth":
        # loss = log(|z^ - z|) = log(|\noise|)
        # grad = sign(\noise) / |noise|
        output = np.sign(input) / (np.abs(input) + 0.3)
    elif name == "adabins":
        z_max = 50
        beta = 25
        output = beta*np.sign(input) / np.power(beta + np.abs(input), 2.0)
        # term1  = np.log(1 + z_max/np.abs(input))
        # term2  = (z_max + 2*np.abs(input))/ np.power(z_max + np.abs(input), 2.0)
        # output = np.sign(input)/z_max *(term1 - term2)
    elif name == "dice":
        index = input <= length
        output[index] = np.sign(input[index]) / length
    elif name == "iou":
        index = input <= length
        output[index] = 2*length*np.sign(input[index]) / np.power(length + np.abs(input[index]), 2.0)
    else:
        raise NotImplementedError
    return output

def var_gradient_sq(input, length= 4, name= "l2"):
    return np.mean(np.power(gradient(input, length= length, name= name), 2.0))

def theory_bound(noise_std= 1, length= 4, name= "l2"):
    if name == "l1":
        bound = 1
    elif name == "l2":
        bound = noise_std*noise_std
    elif name == "smoothL1":
        bound = 1 - np.sqrt(1.0/(15 *np.pi *np.e))
    elif name == "dice":
        bound = (1.0/(length*length)) * erf(length/(math.sqrt(2) * noise_std))
    elif name == "iou_v0":
        bound = 16*np.sqrt(2.0)/(np.sqrt(7)*np.sqrt(np.pi)*length * noise_std)
    elif name == "iou":
        sigma_1 = 5
        y_0 = (4.0/(length*length))
        y_1 = 0#(1.0/(3*length*length*length))
        bound = y_0 - (y_0 - y_1)*noise_std/sigma_1
    return bound
# ==================================================================================================
# Mains starts here
# ==================================================================================================
N              = 100000
noise_mean     = 0.0
noise_std_arr = np.arange(0.02, 3.1, 0.05)#np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0])
num_std = len(noise_std_arr)
num_cat = 2

lw = params.lw + 2
ms = 0#params.ms
dashes = params.dashes

c1 = params.color_seaborn_5
c2 = params.color_seaborn_2
c3 = params.color_seaborn_15
c4 = params.color_seaborn_4
cat_color = ["dodgerblue", params.color_seaborn_1]

#c1 = params.color_set1_pink/255
#c2 = params.color_set1_cyan/255
#c3 = params.color_seman_building/255
#c4 = params.color_set2_pink_light/255
#cat_color = [params.color_seman_blue/255, params.color_set1_yellow/255]
len_cat_arr = np.array([4, 12])
name_cat_arr= ['Car', 'Trailer']
length      = len_cat_arr[0]

th_lw     = lw
legend_fs = params.legend_fs + 8
legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad + 0.1
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing

# 0 --> empirical, 1 --> theoretical
L1_var  = np.zeros((2, num_std))
L2_var  = np.zeros((2, num_std))
Sm_var  = np.zeros((2, num_std))
Adabins_var = np.zeros((2, num_std))
log_depth_var = np.zeros((2, num_std))
Iou_var_old = np.zeros((num_cat, 2, num_std))
Iou_var     = np.zeros((num_cat, 2, num_std))
Dic_var     = np.zeros((num_cat, 2, num_std))


# Sample points
for i, noise_std in enumerate(noise_std_arr):
    np.random.seed(0)
    noise = np.random.normal(noise_mean, noise_std, N)

    # Experimental bounds
    L1_var [0, i] = var_gradient_sq(input= noise, length= length, name= "l1")
    L2_var [0, i] = var_gradient_sq(input= noise, length= length, name= "l2")
    Sm_var [0, i] = var_gradient_sq(input= noise, length= length, name= "smoothL1")
    log_depth_var [0, i] = var_gradient_sq(input= noise, length= length, name= "log_depth")
    Adabins_var[0, i] = var_gradient_sq(input= noise, length= length, name="adabins")
    for j, len_cat in enumerate(len_cat_arr):
        Dic_var[j, 0, i] = var_gradient_sq(input= noise, length= len_cat, name= "dice")
        Iou_var[j, 0, i] = var_gradient_sq(input= noise, length= len_cat, name= "iou")

    #Theoretical bounds
    L1_var [1, i] = theory_bound(noise_std, length= length, name= "l1")
    L2_var [1, i] = theory_bound(noise_std, length= length, name= "l2")
    Sm_var [1, i] = theory_bound(noise_std, length= length, name= "smoothL1")
    for j, len_cat in enumerate(len_cat_arr):
        Iou_var[j, 1, i]     = theory_bound(noise_std, length= len_cat, name= "iou")
        Iou_var_old[j, 1, i] = theory_bound(noise_std, length= len_cat, name= "iou_v0")

# ==============================================================================
# Full Figure
# ==============================================================================
plt.figure(figsize= params.size, dpi= params.DPI)
# Empirical plots
plt.plot(noise_std_arr, L1_var     [0], '-o', markersize= ms-4, c= c1, label= r'$L_1$'       , linewidth= lw, zorder= -1)
plt.plot(noise_std_arr, L2_var     [0], '-o', markersize= ms-4, c= c2, label= r'$L_2$'       , linewidth= lw)
plt.plot(noise_std_arr, Sm_var     [0], '-o', markersize= ms-4, c= c3, label= r'$L_{smooth}$', linewidth= lw)
plt.plot(noise_std_arr, log_depth_var[0], '-o', markersize= ms-4, c= c4, label= r'$L_{log diff}$', linewidth= lw, zorder= -2)
# plt.plot(noise_std_arr, Adabins_var[0], '-o', markersize= ms-4, c= c4, label= r'$L_{adabin}$', linewidth= lw)
for j, name_cat in enumerate(name_cat_arr):
    plt.plot(noise_std_arr, Dic_var[j, 0], '-o', markersize= ms-4, c= cat_color[j], label= r'$L_{dice}$ ' + name_cat + r' $(\ell=$' + str(len_cat_arr[j]) + 'm)', linewidth= lw)
#for j, name_cat in enumerate(name_cat_arr):
#    plt.plot(noise_std_arr, Iou_var[j, 0], '--', markersize= ms-4, c= cat_color[j], label= r'$L_{iou}$  ' + name_cat + r' $(\ell=$' + str(len_cat_arr[j]) + 'm)', linewidth= lw-2)

# Theoretical plots
# plt.plot(noise_std_arr, L1_var [1], '--x', markersize= ms , c= c1, label= r'$L_1$'       , linewidth= lw)
# plt.plot(noise_std_arr, L2_var [1], '--x', markersize= ms , c= c2, label= r'$L_2$'       , linewidth= lw)
# plt.plot(noise_std_arr, Sm_var [1], '--', markersize= ms , c= c3, label= r'$L_{sm}$ Th'  , linewidth= lw)
# for j, name_cat in enumerate(name_cat_arr):
    # plt.plot(noise_std_arr, Iou_var[j, 1]    , '--', dashes= dashes, markersize= ms , c= cat_color[j], label= r'$L_{iou}$ ' + name_cat + ' Th', linewidth= lw)
    # plt.plot(noise_std_arr, Iou_var_old[j, 1], '--', dashes= dashes, markersize= ms , c= cat_color[j], label= r'$L_{dice}$ ' + name_cat + ' Th', linewidth= lw)

#plt.axvspan(1, np.max(noise_std_arr)+0.1, facecolor= '0.6', alpha=0.1)

plt.ylabel('log Var' + r'($\epsilon$) $\downarrow$')
plt.xlabel('Noise std' + r'($\sigma$)')
plt.xlim(np.min(noise_std_arr)-0.02,np.max(noise_std_arr)+0.1)
plt.xticks(np.arange(0, np.max(noise_std_arr), 0.5), np.arange(0, np.max(noise_std_arr), 0.5))
plt.grid(True)
plt.yscale('log')
plt.ylim(bottom=10**-4.2)
plt.legend(loc= "lower right", ncol= 2, fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, pad_inches= 0.05, path= os.path.join("images", "convergence_analysis.png"))
# plt.show()
plt.close()





# ==============================================================================
# Teaser Figure
# ==============================================================================
plt.figure(figsize= params.size, dpi= params.DPI)
# Empirical plots
plt.plot(noise_std_arr, L1_var     [0], '-o', markersize= ms-4, c= c1, label= r'$L_1$'       , linewidth= lw, zorder= -1)
plt.plot(noise_std_arr, L2_var     [0], '-o', markersize= ms-4, c= c2, label= r'$L_2$'       , linewidth= lw)
#plt.plot(noise_std_arr, Sm_var     [0], '-o', markersize= ms-4, c= c1, label= r'$L_{smooth}$', linewidth= lw)
#plt.plot(noise_std_arr, log_depth_var[0], '-o', markersize= ms-4, c= c4, label= r'$L_{log diff}$', linewidth= lw, zorder= -2)
# plt.plot(noise_std_arr, Adabins_var[0], '-o', markersize= ms-4, c= c4, label= r'$L_{adabin}$', linewidth= lw)
for j, name_cat in enumerate(name_cat_arr):
    plt.plot(noise_std_arr, Dic_var[j, 0], '-o', markersize= ms-4, c= cat_color[j], label= r'$L_{dice}$ ' + name_cat + r' $(\ell=$' + str(len_cat_arr[j]) + 'm)', linewidth= lw)
# for j, name_cat in enumerate(name_cat_arr):
#     plt.plot(noise_std_arr, Iou_var[j, 0], '--', markersize= ms-4, c= cat_color[j], label= r'$L_{iou}$  ' + name_cat + r' $(\ell=$' + str(len_cat_arr[j]) + 'm)', linewidth= lw)

plt.ylabel('log Var' + r'($\epsilon$) $\downarrow$')
plt.xlabel('Noise std' + r'($\sigma$)')
plt.xlim(np.min(noise_std_arr)-0.02,np.max(noise_std_arr)+0.1)
plt.xticks(np.arange(0, np.max(noise_std_arr), 0.5), np.arange(0, np.max(noise_std_arr), 0.5))
plt.grid(True)
plt.yscale('log')
plt.ylim(bottom=10**-4.2)
plt.legend(loc= "lower right", ncol= 2, fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
savefig(plt, pad_inches= 0.05, path= os.path.join("images", "convergence_analysis_teaser.png"))
# plt.show()
