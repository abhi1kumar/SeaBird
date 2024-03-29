
import os, sys
sys.path.append(os.getcwd())

import numpy as np
import plot.plotting_params as params
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage import img_as_ubyte
import imageio

# from lib.util import get_correlation, combine
# from lib.math_3d import project_3d_points_in_4D_format, project_3d_corners
# from lib.core import iou
# from lib.core import iou3d
from panoptic_bev.helpers.file_io import read_csv

sub_index = 1

def savefig(plt, path, show_message= True, tight_flag= True, pad_inches= 0, newline= True):
    if show_message:
        print("Saving to {}".format(path))
    if tight_flag:
        plt.savefig(path, bbox_inches='tight', pad_inches= pad_inches)
    else:
        plt.savefig(path)
    if newline:
        print("")

def diverge_map(top_color_frac=(1.0, 0.0, 0.25), bottom_color_frac=(1.0, 0.906, 0.04)):
    '''
    top_color_frac and bottom_color_frac are colors that will be used for the two
    ends of the spectrum.
    Reference
    https://towardsdatascience.com/creating-colormaps-in-matplotlib-4d4de78a04b8
    '''
    # import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    N = 128
    top_color = np.ones((N, 4))
    top_color[:, 0] = np.linspace(top_color_frac[0], 1, N) # R
    top_color[:, 1] = np.linspace(top_color_frac[1], 1, N) # G
    top_color[:, 2] = np.linspace(top_color_frac[2], 1, N) # B
    top_color_cmp = ListedColormap(top_color)

    if bottom_color_frac is not None:
        bottom_color = np.ones((N, 4))
        bottom_color[:, 0] = np.linspace(bottom_color_frac[0], 1, N)
        bottom_color[:, 1] = np.linspace(bottom_color_frac[1], 1, N)
        bottom_color[:, 2] = np.linspace(bottom_color_frac[2], 1, N)
        bottom_color_cmp   = ListedColormap(bottom_color)

        newcolors2 = np.vstack((bottom_color_cmp(np.linspace(0, 1, 128)), top_color_cmp(np.linspace(1, 0, 128))))
    else:
        newcolors2 = top_color_cmp(np.linspace(1, 0, 128))
    double = ListedColormap(newcolors2, name='double')

    return double


def parse_predictions(folder, before= True, return_score= True):
    if before:
        file = "predictions_bef_nms.npy"
    else:
        file = "predictions.npy"
    pred_path       = os.path.join(folder, file)
    print("Loading {}...".format(pred_path))
    predictions     = np.load(pred_path)

    # Drop the label column and convert the numpy array to float
    predictions     = predictions[:, sub_index:].astype(float)
    score           = predictions[:, 15 - sub_index]
    h2d_pred_all    = predictions[:, 17 - sub_index]
    h2d_general_all = predictions[:, 18 - sub_index]
    h2d_special_all = predictions[:, 19 - sub_index]

    return h2d_pred_all, h2d_general_all, score

def get_output_file_path(prefix, postfix="", relative= False, threshold= 0, before_nms_flag= True, folder= None):
    output_image_file = prefix
    if relative:
        output_image_file += '_rel'
    output_image_file += postfix

    if before_nms_flag:
        output_image_file += '_on_all_predictions'

    if threshold > 0:
        output_image_file += '_class_conf_gt_' + str(threshold)

    output_image_file += '.png'

    if folder is not None:
        path = os.path.join(folder, output_image_file)

    return path

def throw_samples(x, y, frac_to_keep= 0.9, show_message= True, throw_around_center= True):
    """
        Throws the outliers based on sorting
        frac = 0.75 suggests keep the first three quarters of the data and throw
        away the remaining data. Keep the elements belonging to the valid data
    """
    if show_message:
        print("Using {:.2f}% of the data".format(frac_to_keep * 100.0))
    samples_to_keep = int(x.shape[0] * frac_to_keep)

    # Sort the x array and get the indices
    sorted_indices = np.abs(x).argsort()
    # Keep the indices which are required
    if throw_around_center:
        center_index = sorted_indices.shape[0]//2
        keep_index   = sorted_indices[(center_index-samples_to_keep//2) : (center_index + samples_to_keep//2)+1]
    else:
        keep_index   = sorted_indices[0:samples_to_keep]

    # Use the same index for x and y
    x = x[keep_index]
    y = y[keep_index]

    return x, y

def get_bins(x, num_bins):
    # sort the data
    x_min = np.min(x)
    x     = np.sort(x)

    pts_per_bin = int(np.ceil(x.shape[0]/(num_bins)))
    # print("Num bins= {}, pts per bin  = {}".format(num_bins, pts_per_bin))
    # bins contain a lower bound. so 1 extra element
    bins  = np.zeros((num_bins+1, ))

    bins[0] = x_min
    for i in range(1,bins.shape[0]):
        if i*pts_per_bin < x.shape[0]:
            end_ind = i*pts_per_bin
        else:
            end_ind = x.shape[0]-1
        bins[i] = x[end_ind]

    return bins

def draw_rectangle(ax, rect_x_left, rect_y_left, rect_width, rect_height, img_width= 100, img_height= 100, edgecolor= 'r', linewidth= params.lw):
    """
    Draw a rectangle on the image
    :param ax:
    :param img_width:
    :param img_height:
    :param rect_x_left:
    :param rect_y_left:
    :param rect_width:
    :param rect_height:
    :param angle:
    :return:
    """

    x = rect_x_left
    y = rect_y_left
    # Rectangle patch takes the following coordinates (xy, width, height)
    # Reference https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle
    # :                xy------------------+
    # :                |                  |
    # :              height               |
    # :                |                  |
    # :                ------- width -----+
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width= rect_width, height= rect_height, linewidth= linewidth, edgecolor= edgecolor, facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    return ax

def draw_circle(ax, x, y, radius= 6, color= 'r', edgecolor= None):
    circle1 = plt.Circle((x, y), radius, color= color)
    ax.add_artist(circle1)

def draw_line(plt, x1, y1, x2, y2, color= 'b', marker= 'o'):
    # :                ------- width (x) -----+
    # :                |
    # :              height (y)
    # :                |
    # :                +
    plt.plot(x1, y1, x2, y2, color= color, linewidth= 100, marker= marker, markerfacecolor= color)

def visualize_semantic_map(bev_semantic):
    # bev_semantic as single channel image
    # Semantic KITTI color codes
    # https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    semantic_color_map = {0: [255, 255, 255],
    1: [255, 0, 0],      # Wall 3
    10: [100, 150, 245], # Car 9
    11: [100, 230, 245], # Rider 8
    13: [100, 80, 250],  # Truck 10
    30: [255, 30, 30],   # Person 7
    40: [255, 59, 141],  # Road 0
    48: [75, 0, 75],     # Sidewalk 1
    50: [255, 200, 0],   # Terrain 5
    51: [255, 120, 50],  # Occluded 6
    70: [0, 175, 0],     # Vegetation 4
    72: [150, 240, 80],
    1000: [47, 79, 79],  # Building 2
    }
    panoptic_bev_id_to_semantic_kitti_id_mapping = {255: 0, 0: 40, 1: 48, 2: 1000, 3: 1, 4: 70, 5: 50, 6: 51, 7: 30, 8: 11, 9: 10, 10: 13}

    cat_img = panoptic_bev_id_to_semantic_kitti_id_mapping.keys()
    bev_semantic_new = np.zeros((bev_semantic.shape[0]*bev_semantic.shape[1], 3)).astype(np.uint8)
    bev_semantic_old = bev_semantic.flatten()

    for i, cat_curr in enumerate(cat_img):
        semantic_kitti_color_id = panoptic_bev_id_to_semantic_kitti_id_mapping[cat_curr]
        bev_semantic_new[bev_semantic_old == cat_curr, :] = semantic_color_map[semantic_kitti_color_id]
    bev_semantic_new = bev_semantic_new.reshape((bev_semantic.shape[0], -1, 3))

    return bev_semantic_new

def get_left_point_width_height(gts):
    rect_x_left  = gts[0]
    rect_y_left  = gts[1]
    rect_x_right = gts[2]
    rect_y_right = gts[3]
    rect_width   = rect_x_right - rect_x_left
    rect_height  = rect_y_right - rect_y_left

    return rect_x_left, rect_y_left, rect_width, rect_height

def open_gif_writer(file_path, duration= 0.5):
    print("=> Saving to {}".format(file_path))
    gif_writer = imageio.get_writer(file_path, mode='I', duration= duration)

    return gif_writer

def convert_fig_to_ubyte_image(fig):
    canvas = FigureCanvas(fig)
    # draw canvas as image
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    image = img_as_ubyte(image)

    return image

def add_ubyte_image_to_gif_writer(gif_writer, ubyte_image):

    gif_writer.append_data(ubyte_image)

def close_gif_writer(gif_writer):

    gif_writer.close()

def draw_red_border(sub1):
    autoAxis = sub1.axis()
    rec = patches.Rectangle((autoAxis[0],autoAxis[2]),(autoAxis[1]-autoAxis[0]),(autoAxis[3]-autoAxis[2]),fill=False, color= 'red', lw= 7)
    rec = sub1.add_patch(rec)
    rec.set_clip_on(False)
