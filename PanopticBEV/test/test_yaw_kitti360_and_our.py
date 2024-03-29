"""
    Sample Run:
    python test/test_yaw_kitti360_and_our.py
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

def get_kitti_360_yaw(rot_pts):
    return -1 * np.arctan2(rot_pts[2,0]-rot_pts[0,0], rot_pts[2,1]-rot_pts[0,1])

def get_our_yaw(rot_pts):
    #                     +Y
    # 5 (-0.5l,  0.5w)    |             0 (0.5l,  0.5w)
    #          ===========|===========
    #          |          |           |
    #     -----|----------|-----------|-----------+ X
    #          |          |           |
    #          ===========|===========
    # 7 (-0.5l, -0.5w)    |            2 (0.5l, -0.5w)
    #                     |
    midpoint = 0.5*(rot_pts[2] + rot_pts[0])
    return np.arctan2(midpoint[1], midpoint[0])

# Dummy car
l = 4
w = 2
h = 2
# Corners are in the KITTI-360 annotation format.
x_corners = [l/2, l/2, l/2, l/2,-l/2,-l/2,-l/2,-l/2]
y_corners = [w/2, w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2]
z_corners = [h/2,-h/2, h/2,-h/2,-h/2, h/2,-h/2, h/2]

pts          = np.stack((x_corners, y_corners), axis=1)

rotate_angle = np.arange(-1, 1, 0.01)*np.pi
our_yaws     = np.zeros_like(rotate_angle)
kitti_yaws   = np.zeros_like(rotate_angle)

for i, t in enumerate(rotate_angle):
    c = np.cos(t)
    s = np.sin(t)
    # I assume the object frame of reference is X right, Z up and Y inside
    # The rotation matrix rot_z is from
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L186-L191
    #
    # Note that rot_z transforms a point (l/2,0) to (0,l/2) for \theta = 90.
    #                    + Y
    #                    |
    #                    |
    #                    |        (l/2, 0)
    #                    |********-----------+ X
    #
    #                    + Y
    #                    |
    #                    * (0,l/2)
    #                    *
    #                    *
    #                    *-------------------+ X
    # So, it is ANTI-clockwise rotation about z-axis.
    # and not clockwise about z-axis as commented here
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/evaluation/semantic_3d/evalDetection.py#L543
    rot_z   = np.array([[c, -s], [s, c]])

    # Rotate points
    rot_pts       = np.matmul(rot_z, pts.T).T
    kitti_yaws[i] = get_kitti_360_yaw(rot_pts)
    our_yaws  [i] = get_our_yaw(rot_pts)

lw = 4
plt.figure(figsize=(10,6), dpi= 200)
plt.plot(rotate_angle, rotate_angle, label= 'Ideal', lw= lw, c= 'k')
plt.plot(rotate_angle, kitti_yaws  , label= 'KITTI', lw= lw)
plt.plot(rotate_angle, our_yaws    , label= 'Ours' , lw= lw-2)
plt.legend(loc='lower right')
plt.grid(True)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$\pi$', r'$-\pi/2$', 0, r'$\pi/2$', r'$\pi$'])
plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$\pi$', r'$-\pi/2$', 0, r'$\pi/2$', r'$\pi$'])
plt.xlabel('GT   Rotation (rad)')
plt.ylabel('Pred Rotation (rad)')
plt.savefig('images/yaw_kitti_vs_ours.png', bbox_inches='tight')
plt.show()
plt.close()
