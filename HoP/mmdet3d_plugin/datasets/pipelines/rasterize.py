"""
Taken from
https://github.com/zhangyp15/BEVerse/blob/main/projects/mmdet3d_plugin/datasets/pipelines/rasterize.py
"""
import torch
import numpy as np
import cv2
import copy

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.bbox.structures.coord_3d_mode import Coord3DMode
from ..utils import preprocess_map
import pdb

import warnings
warnings.filterwarnings('ignore')

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


@PIPELINES.register_module()
class RasterizeMapVectors(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,
                 map_grid_conf=None,
                 map_max_channel=3,
                 map_thickness=5,
                 map_angle_class=36,
                 for_classes=False,
                 seg_style='only_car',
                 ignore_index=255,
                 filter_invisible=True
                 ):

        self.map_max_channel = map_max_channel
        self.map_thickness = map_thickness
        self.map_angle_class = map_angle_class

        map_xbound, map_ybound = map_grid_conf['xbound'], map_grid_conf['ybound']

        # patch_size: 在 y, x 方向上的坐标 range
        patch_h = map_ybound[1] - map_ybound[0]
        patch_w = map_xbound[1] - map_xbound[0]

        # canvas_size: 在 y, x 方向上的 bev 尺寸
        canvas_h = int(patch_h / map_ybound[2])
        canvas_w = int(patch_w / map_xbound[2])

        self.map_patch_size = (patch_h, patch_w)
        self.map_canvas_size = (canvas_h, canvas_w)
        self.for_classes     = for_classes
        self.grid_conf       = map_grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            self.grid_conf['xbound'], self.grid_conf['ybound'], self.grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution     = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension      = self.bev_dimension.numpy()
        self.spatial_extent     = (self.grid_conf['xbound'][1], self.grid_conf['ybound'][1])
        self.ignore_index       = ignore_index
        self.seg_style          = seg_style
        self.filter_invisible   = filter_invisible

        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']
        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])
        self.all_cls_ids = np.arange(len(nusc_classes))

        if self.seg_style == 'only_car':
            self.num_cls = 1
        elif self.seg_style == 'car_and_vehicle':
            # Car segmentation on nuScenes refers to all bounding boxes of class vehicle.car and
            # vehicle segmentation on nuScenes refers to all bounding boxes of meta-category
            # vehicle.
            # References:
            # [1] Lift Splat Shoot: Jonah Philion Sanja Fidler, ECCV 2020, https://arxiv.org/pdf/2008.05711.pdf
            # [2] https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/color_map.py
            self.num_cls = 2
        elif self.seg_style == 'car_vehicle_barrier_pedestrian_cone':
            self.num_cls = 5
        else:
            self.num_cls = 1
        self.bev_map_size = (self.num_cls, self.bev_dimension[1], self.bev_dimension[0])
        self.coord_system = "lidar"  # or 'camera'
        self.debug = False

    def __call__(self, results):
        if not self.for_classes:
            vectors = results['vectors']
            for vector in vectors:
                vector['pts'] = vector['pts'][:, :2]

            semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
                vectors, self.map_patch_size, self.map_canvas_size, self.map_max_channel, self.map_thickness, self.map_angle_class)

            num_cls = semantic_masks.shape[0]
        else:
            # Sample foreground classes
            # annotation_token ==> instance_id
            instance_map = {}

            # convert LiDAR bounding boxes to motion labels
            num_frame = len(results['gt_bboxes_3d'])
            all_gt_bboxes_3d = results['gt_bboxes_3d']
            all_gt_labels_3d = results['gt_labels_3d']
            # 4x4 transformation matrix (if exist)
            bev_transform = results.get('aug_transform', None)

            segmentations = []
            instances = []

            # 对于 invalid frame: 所有 label 均为 255
            # 对于 valid frame: seg & instance 背景是 0，其它背景为255
            for frame_index in range(1):
                car_cnt = 0
                # Compared to BEVerse, HoP only sends the current frame boxes
                gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d, all_gt_labels_3d

                if gt_bboxes_3d is None:
                    # for invalid samples
                    segmentation = np.ones(self.bev_map_size) * self.ignore_index
                else:
                    # for valid samples
                    segmentation = np.zeros(self.bev_map_size)

                    if self.seg_style == 'only_car' or self.seg_style == 'car_and_vehicle':
                        vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                        gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                        gt_labels_3d = gt_labels_3d[vehicle_mask]

                    # valid sample and has objects
                    if len(gt_bboxes_3d.tensor) > 0:
                        if self.debug:
                            print(gt_bboxes_3d.corners.numpy().shape)
                        num_valid_boxes = gt_labels_3d.shape[0]

                        if self.coord_system == "lidar":
                            # torch.Tensor: Coordinates of corners of all the boxes
                            # in shape (N, 8, 3).
                            #
                            # Convert the boxes to corners in clockwise order, in form of
                            # ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``
                            #
                            # .. code-block:: none
                            #
                            #                                    up z
                            #                     front x           ^
                            #                          /            |
                            #                         /             |
                            #           (x1, y0, z1) + -----------  + (x1, y1, z1)
                            #                       /|            / |
                            #                      / |           /  |
                            #      1 (x0, y0, z1) + --4-------- +   + 7(x1, y1, z0)
                            #                     |  /      .   |  /
                            #                     | / origin    | /
                            #     left y<-------- + ----------- + 3 (x0, y1, z0)
                            #         0 (x0, y0, z0)
                            # Reference:
                            # https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/structures/bbox_3d/lidar_box3d.py#L42-L58
                            bbox_corners_meter = gt_bboxes_3d.corners[:, [
                                0, 3, 7, 4], :2].numpy()

                        elif self.coord_system == "camera":
                            # Convert in camera coordinate system so that BEV map looks reasonable.
                            gt_bboxes_3d = copy.deepcopy(gt_bboxes_3d).convert_to(dst=Coord3DMode.CAM)

                            # torch.Tensor: Coordinates of corners of all the boxes
                            # in shape (N, 8, 3).
                            #
                            # Convert the boxes to  in clockwise order, in the form of
                            # (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)
                            #
                            # .. code-block:: none
                            #
                            #                  front z
                            #                       /
                            #                      /
                            #        (x0, y0, z1) + -----------  + (x1, y0, z1)
                            #                    /|            / |
                            #                   / |           /  |
                            #     (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                            #                  |  /      .   |  /
                            #                  | / origin    | /
                            #    3(x0, y1, z0) + ----------- + -------> x right
                            #                  |             7(x1, y1, z0)
                            #                  |
                            #                  v
                            #             down y
                            # Reference https://github.com/Sense-X/HoP/blob/9f6e8823c26a7b126091d6dffaf7eb277a14b5d5/mmdet3d/core/bbox/structures/cam_box3d.py
                            bbox_corners_meter = gt_bboxes_3d.corners[:, [
                                3, 7, 6, 2], :][:, :, [2, 0]].numpy()
                        bbox_corners = np.round(
                            (bbox_corners_meter - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)

                        for index in range(num_valid_boxes):
                            poly_region = bbox_corners[index]
                            if self.seg_style == 'only_car':
                                cv2.fillPoly(segmentation[0], [poly_region], 1.0)
                                car_cnt += 1
                            elif self.seg_style == 'car_and_vehicle':
                                # self.vehicle_cls_ids = 0,3,2,7,6,1,4 where first one is Car
                                if gt_labels_3d[index] == 0:
                                    cv2.fillPoly(segmentation[0], [poly_region], 1.0)
                                else:
                                    cv2.fillPoly(segmentation[1], [poly_region], 2.0)
                            elif self.seg_style == 'car_vehicle_barrier_pedestrian_cone':
                                # self.vehicle_cls_ids = 0,3,2,7,6,1,4 where first one is Car
                                if gt_labels_3d[index] == 0:
                                    # car
                                    if not self.debug:
                                        cv2.fillPoly(segmentation[0], [poly_region], 1.0)
                                    else:
                                        car_cnt += 1
                                        cv2.fillPoly(segmentation[0], [poly_region], car_cnt)
                                elif gt_labels_3d[index] == 5:
                                    # barrier
                                    cv2.fillPoly(segmentation[2], [poly_region], 3.0)
                                elif gt_labels_3d[index] == 8:
                                    # pedestrian
                                    cv2.fillPoly(segmentation[3], [poly_region], 4.0)
                                elif gt_labels_3d[index] == 9:
                                    # cone
                                    cv2.fillPoly(segmentation[4], [poly_region], 5.0)
                                else:
                                    # vehicle
                                    cv2.fillPoly(segmentation[1], [poly_region], 2.0)

                segmentations.append(segmentation)

            # segmentation = 1 where objects are located
            semantic_indices = np.stack(segmentations, axis=0).astype(np.uint8)[0] # Cls x H x W
            semantic_masks   = np.full(semantic_indices.shape, False)              # Cls x H x W
            semantic_masks[semantic_indices > 0] = True
            semantic_indices = np.max(semantic_indices, axis=0)                    # H x W

        if self.debug:
            from matplotlib import pyplot as plt
            print(car_cnt)
            car_mask = (gt_labels_3d == 0).numpy()
            gt_bboxes_car          = gt_bboxes_3d.tensor[car_mask].numpy()
            bbox_corners_meter_car = bbox_corners_meter[car_mask]
            bbox_corners_car       = bbox_corners[car_mask]
            plt.imshow(semantic_indices, vmin= 0, vmax= car_cnt, cmap='Reds')
            plt.show(block=False)

        indices = np.arange(1, self.num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = np.sum(semantic_masks * indices, axis=0)

        results.update({
            'semantic_map': torch.from_numpy(semantic_masks),              # Cls x H x W
            'semantic_indices': torch.from_numpy(semantic_indices).long(), # H x W
        })

        return results
