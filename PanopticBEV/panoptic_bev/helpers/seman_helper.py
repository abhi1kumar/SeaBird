import os
import numpy as np
import copy
import cv2
import glob
import shutil

from panoptic_bev.helpers.kitti_360_evalDetection import evaluate_kitti_360_verbose
from panoptic_bev.helpers.kitti_360_evalDetection_windows import evaluate_kitti_360_windows_verbose
from panoptic_bev.helpers.file_io import read_csv, read_lines, write_lines

def bev_px_to_metric(center_sem_seg, bev_h, bev_w, res_percent= 100):
    """
    centers in (x,z) space
    """
    # Center of the bev is (bev_w/2, bev_h)
    center_sem_final = copy.deepcopy(center_sem_seg)

    center_sem_final[:, 0]  = center_sem_seg[:, 0] - bev_w/2.0
    center_sem_final[:, 1]  = bev_h - center_sem_seg[:, 1]

    # Upsample BEV 3D center (x,z) based on resolution
    center_sem_final *= 100.0/res_percent

    # Multiply by scaling factors to convert to metric space
    center_sem_final[:, 0] = center_sem_final[:, 0] * 0.07360504032236664 + 0.8106205618037258
    # center_sem_final[:, 0] /= 18.21
    center_sem_final[:, 1] /= 14.35 # This seems correct

    return center_sem_final

def metric_to_bev_px(center_metric, bev_h, bev_w, res_percent= 100):
    """ center_metric = [[X, Z]] format"""
    center_sem_seg = copy.deepcopy(center_metric)

    if center_sem_seg.ndim == 2:
        center_sem_seg[:, 0] = (center_sem_seg[:, 0] - 0.8106205618037258) / 0.07360504032236664
        center_sem_seg[:, 1] *= 14.35

        # Downsample center_sem_seg
        center_sem_seg /= 100.0/res_percent

        # Finally change the coordinates
        center_sem_seg[:, 0]  += bev_w/2.0
        center_sem_seg[:, 1]   = bev_h - center_sem_seg[:, 1]

    elif center_sem_seg.ndim == 3:
        center_sem_seg[:, :, 0] = (center_sem_seg[:, :, 0] - 0.8106205618037258) / 0.07360504032236664
        center_sem_seg[:, :, 1] *= 14.35

        # Downsample center_sem_seg
        center_sem_seg /= 100.0/res_percent

        # Finally change the coordinates
        center_sem_seg[:, :, 0]  += bev_w/2.0
        center_sem_seg[:, :, 1]   = bev_h - center_sem_seg[:, :, 1]

    return center_sem_seg

def box_to_params(box_full_res, bev_h =704, bev_w = 768):
    # Converts axis aligned BEV boxes to 3D params X,Z,l,w,yaw
    # box_full_res = 4 x 2 array in (x, z) format with order as
    # [bl (x, z)
    #  tl (x, z)
    #  br (x, z)
    #  lr (x, z)] format
    # The angle is a reduced yaw (range 0 to 90)
    # Reference: https://theailearner.com/tag/cv2-minarearect/

    # We first scale the pixel coordinates before passing on to cv2.minAreaRect since
    # cv2.minAreaRect needs integer coordinates.
    # https://theailearner.com/tag/cv2-minarearect/
    SCALE = 1000
    box_scaled = np.asarray(box_full_res * SCALE, np.int64)
    center, dim, angle_deg = myMinAreaRect(box_scaled)
    x3d_px, z3d_px     = np.array(center, np.float32)/SCALE
    w3d_px, l3d_px   = np.array(dim, np.float32)/SCALE

    center_sem_seg   = np.array([[x3d_px, z3d_px], [0, 0], [l3d_px, 0], [0.0, w3d_px]])
    temp_out         = bev_px_to_metric(center_sem_seg, bev_h= bev_h, bev_w= bev_w)
    x3d_box, z3d_box = temp_out[0]
    # Affine transform preserves distances
    l3d_box  = np.linalg.norm(temp_out[1] - temp_out[2])
    w3d_box  = np.linalg.norm(temp_out[1] - temp_out[3])

    # Convert angle to radians
    angle = angle_deg * np.pi/180

    if l3d_box < w3d_box:
        w3d_box, l3d_box = l3d_box, w3d_box

    return x3d_box, z3d_box, l3d_box, w3d_box, angle

def myMinAreaRect(box):
    box    = box.astype(np.float64)
    # Calculate center and center the box
    center = np.mean(box, axis= 0)
    box   -= center[np.newaxis, :]

    # First figure out the bl, tl, br, tr points
    # Points on left and right  of center
    left  = np.where(box[:, 0] <= 0)[0]
    right = np.where(box[:, 0] >= 0)[0]

    if left.shape[0] > 2 and right.shape[0]  > 2:
        # 2 points lie on x-axis
        intersection = np.intersect1d(left, right)
        left = np.union1d(np.setdiff1d(left , intersection), intersection[0])
        right= np.union1d(np.setdiff1d(right, intersection), intersection[1])

    bl    = box[left[np.argmin(box[left, 1])] ]
    tl    = box[left[np.argmax(box[left, 1])] ]
    br    = box[right[np.argmin(box[right, 1])] ]
    tr    = box[right[np.argmax(box[right, 1])] ]

    peri_line_center = np.array([0.5*(bl + tl), 0.5*(tl + tr), 0.5*(tr + br), 0.5*(br + bl)])
    dist_peri        = np.linalg.norm(peri_line_center, 2, axis=1)

    l3d              = np.max(dist_peri) * 2.0
    w3d              = np.min(dist_peri) * 2.0
    dim              = np.array([w3d, l3d])

    l3d_index        = np.argmax(dist_peri)
    tt = 0.5*(tl + tr)
    rr = 0.5*(tr + br)

    # Axes convention
    #        |
    #        |
    # -------|---|----+ x
    #        | \| kitti_angle
    #        |  \
    #        |   +
    #        +
    #        z


    if l3d_index == 0 or l3d_index == 2:
        # maximum along left/right
        vector = rr
    else:
        # maximum along top/bottom
        vector = tt
    kitti_angle = np.arctan2(vector[1], vector[0])
    angle       = - kitti_angle
    angle_deg   = angle * 180 / np.pi

    return center, dim, angle_deg

def get_obj_level(height, truncation, occlusion):

    if truncation == -1:
        level_str = 'DontCare'
        return 0

    if height >= 40 and truncation <= 0.15 and occlusion <= 0:
        level_str = 'Easy'
        return 1  # Easy
    elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
        level_str = 'Moderate'
        return 2  # Moderate
    elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
        level_str = 'Hard'
        return 3  # Hard
    else:
        level_str = 'UnKnown'
        return 4
