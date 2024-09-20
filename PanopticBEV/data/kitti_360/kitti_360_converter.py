"""
    Sample Run:
    python data/kitti_360/kitti_360_converter.py

    Version 1 Abhinav Kumar
    Converts 3D boxes GT of KITTI-360 dataset to  KITTI format. The pose in KITTI-360 is a 3 dimensional
    angle, and therefore, we only choose the yaw angle for the KITTI format.
"""
import os, sys
sys.path.append(os.getcwd())

from collections import defaultdict
import xml.etree.ElementTree as ET

from panoptic_bev.helpers.kitti_360_util import get_intrinsics, get_ego_to_camera, KITTI360Bbox3D, local2global, get_kitti_style_ground_truth
from panoptic_bev.helpers.file_io import imread, write_lines

from plot.common_operations import *

# detect_cat_list = ['road', 'sidewalk', 'building', 'garage', 'wall', 'fence', 'gate',\
#                       'smallpole', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',\
#                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',\
#                       'bicycle']

# detect_cat_list = ['building', 'garage', 'gate', 'smallpole', 'pole', 'traffic light',
#                       'traffic sign', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
#                       'motorcycle', 'bicycle']

detect_cat_list = ['person', 'car', 'truck', 'bicycle', 'building']
detect_cat_map_dict = {x:x.title() for x in detect_cat_list}
detect_cat_map_dict['person'] = 'Pedestrian'
detect_cat_map_dict['bicycle']= 'Cyclist'

kitti_360_folder    = "data/kitti_360/KITTI-360"
drive_id_list       = [0, 2, 3, 4, 5, 6, 7, 9, 10, 8, 18]
drive_id_test_flag_list = [False]*9 + [True]*2
debug               = False

for drive_id, drive_id_test_flag in zip(drive_id_list, drive_id_test_flag_list):
    drive_name          = "2013_05_28_drive_" + str(drive_id).zfill(4) + "_sync"
    img_folder          = os.path.join(kitti_360_folder, "data_2d_raw", drive_name, "image_00/data_rect")

    camera_calib_folder = os.path.join(kitti_360_folder   , "calibration")
    intrinsics_file     = os.path.join(camera_calib_folder, "perspective.txt")
    ego_to_cam_file     = os.path.join(camera_calib_folder, "calib_cam_to_pose.txt")
    world_to_ego_folder = os.path.join(kitti_360_folder   , "data_poses", drive_name)
    world_to_ego_file   = os.path.join(world_to_ego_folder, "poses.txt")

    output_calib_folder = os.path.join(kitti_360_folder   , "data_2d_raw", drive_name, "calib")
    if not os.path.exists(output_calib_folder):
        os.makedirs(output_calib_folder)

    rect_to_intri, cam_to_rect = get_intrinsics(intrinsic_file_path= intrinsics_file)
    ego_to_camera       = get_ego_to_camera(camera_to_ego_file_path= ego_to_cam_file)

    #===============================================================================================
    # load poses
    #===============================================================================================
    poses  = np.loadtxt(world_to_ego_file)
    # Index Image_Name Mapped Val_Index
    #  303      395      000000009.png
    frames = poses[:,0].astype(np.uint64)
    poses  = np.reshape(poses[:,1:],[-1,3,4])

    #===============================================================================================
    # Make label folder and load 3D bbox annotation
    #===============================================================================================
    if not drive_id_test_flag:
        global_seg_folder   = os.path.join(kitti_360_folder, "data_2d_semantics/train", drive_name, "image_00/instance/")
        label_file          = os.path.join(kitti_360_folder   , "data_3d_bboxes/train", drive_name + ".xml")
        output_label_folder = os.path.join(kitti_360_folder   , "data_2d_raw", drive_name, "label_00")
        if not os.path.exists(output_label_folder):
            os.makedirs(output_label_folder)

        tree = ET.parse(label_file)
        root = tree.getroot()
        objects = defaultdict(dict)
        num_bbox = 0
        global_id_arr = []
        static_id_arr = []
        dynamic_id_arr= []

        for child in root:
            if child.find('transform') is None:
                continue
            obj = KITTI360Bbox3D()
            obj.parseBbox(child)
            # Remove objects which are not of interest, otherwise it increases search time
            if obj.name not in detect_cat_list:
                continue
            globalId = local2global(obj.semanticId, obj.instanceId)
            objects[globalId][obj.timestamp] = obj
            num_bbox += 1
            global_id_arr.append(globalId)

            if obj.timestamp == -1:
                static_id_arr.append(globalId)
            else:
                dynamic_id_arr.append(globalId)

        # Count unique ids
        global_id_arr = np.unique(np.array(global_id_arr))
        static_id_arr = np.unique(np.array(static_id_arr))
        dynamic_id_arr= np.unique(np.array(dynamic_id_arr))

        num_obj    = global_id_arr.shape[0]
        num_static = static_id_arr.shape[0]
        num_dynamic= dynamic_id_arr.shape[0]
        assert(num_dynamic + num_static == num_obj)
    else:
        num_obj     = -1
        num_static  = -1
        num_dynamic = -1
        num_bbox    = -1

    print("=================================================================================================")
    print("{} #valid_frames= {:5d} #obj= {:5d} #sta= {:5d} #dyn= {:5d} #boxes= {:5d} ".format(drive_name, frames.shape[0], num_obj, num_static, num_dynamic, num_bbox))
    print("=================================================================================================")

    for index, frame_id in enumerate(frames):#[107]:#[143]:
        if (index+1) % 1000 == 0 or (index+1) == frames.shape[0]:
            print("{} {:5d} images done.".format(drive_name, index+1), flush= True)

        img_name            = str(frame_id).zfill(10)
        img_path            = os.path.join(img_folder, img_name + ".png")
        if not os.path.exists(img_path):
            continue
        if not drive_id_test_flag:
            global_seg_path     = os.path.join(global_seg_folder, img_name + ".png")
            if not os.path.exists(global_seg_path):
                continue
            output_label_file_path = os.path.join(output_label_folder, img_name + ".txt")
        output_calib_file_path = os.path.join(output_calib_folder, img_name + ".txt")

        #===========================================================================================
        # Calib
        #===========================================================================================
        pose                = poses[frames == frame_id][0]
        world_to_ego        = np.linalg.inv(np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4))))
        world_to_rect       = np.matmul(np.matmul(cam_to_rect, ego_to_camera), world_to_ego)
        camera_calib_final  = np.matmul(rect_to_intri, world_to_rect)
        calib_text_to_write  = 'P2: ' + ' '.join(['{:.12e}'.format(x) for x in rect_to_intri.flatten()[:12].tolist()])
        calib_text_to_write += "\n"   + 'World2Rect: ' + ' '.join(['{:.12e}'.format(x) for x in world_to_rect.flatten()[:12].tolist()]) + "\n"

        # Write calib to the file
        write_lines(output_calib_file_path, calib_text_to_write)

        #===========================================================================================
        # Labels
        #===========================================================================================
        if not drive_id_test_flag:
            img          = imread(img_path)[:,:,::-1]
            h, w, _      = img.shape
            global_seg   = imread(global_seg_path, sixteen_bit= True)

            label_text_to_write = ""

            if debug:
                print(img_name)
                plt.figure()
                ax = plt.gca()
                # plt.imshow(img)
                # Display the instance ids of the object
                plt.imshow(global_seg % 1000)
                plt.title(frame_id)

            #===========================================================================================
            # convert static objects to kitti style GT
            #===========================================================================================
            for static_id in static_id_arr:
                obj       = objects[static_id][-1]
                kitti_cat, truncation, occlusion, alpha, bbox_2d, dimension, center_3d_rect, yaw, proj3d_2d, output_str \
                    = get_kitti_style_ground_truth(obj, camera_calib_final, world_to_rect, detect_cat_list, detect_cat_map_dict, w, h, global_seg)
                if kitti_cat is None:
                    continue
                label_text_to_write += output_str + "\n"

                if debug:
                    u_min, v_min, u_max, v_max = bbox_2d
                    print("Frame ", frame_id, " Static ", static_id, output_str)
                    draw_circle(ax, x= proj3d_2d[0], y= proj3d_2d[1])
                    draw_rectangle(ax, rect_x_left= u_min, rect_y_left= v_min, rect_width= (u_max - u_min), rect_height= (v_max - v_min))
                    plt.text(u_min, v_min, str(static_id))

            #===========================================================================================
            # convert dynamic objects to kitti style GT
            #===========================================================================================
            for dynamic_id in dynamic_id_arr: #[26011]:#
                if frame_id not in objects[dynamic_id].keys():
                    continue

                obj       = objects[dynamic_id][frame_id]
                kitti_cat, truncation, occlusion, alpha, bbox_2d, dimension, center_3d_rect, yaw, proj3d_2d, output_str \
                    = get_kitti_style_ground_truth(obj, camera_calib_final, world_to_rect, detect_cat_list, detect_cat_map_dict, w, h, global_seg)
                if kitti_cat is None:
                    continue
                label_text_to_write += output_str + "\n"

                if debug:
                    u_min, v_min, u_max, v_max = bbox_2d
                    print("Frame ", frame_id, " Dynamic", dynamic_id, output_str)
                    draw_circle(ax, x= proj3d_2d[0], y= proj3d_2d[1])
                    draw_rectangle(ax, rect_x_left= u_min, rect_y_left= v_min, rect_width= (u_max - u_min), rect_height= (v_max - v_min))
                    plt.text(u_min, v_min, str(dynamic_id))

            # Write labels to the file
            # Strip off the last new line
            if len(label_text_to_write) > 0:
                label_text_to_write.strip("\n")
            write_lines(output_label_file_path, label_text_to_write)

        if debug:
            plt.show()
            plt.close()
