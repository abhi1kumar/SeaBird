# Copyright (c) Phigent Robotics. All rights reserved.

# mAP: 0.4108
# mATE: 0.5901
# mASE: 0.2708
# mAOE: 0.5521
# mAVE: 0.2879
# mAAE: 0.2060
# NDS: 0.5147

# Per-class results:
# Object Class        	AP    	ATE   	ASE   	AOE   	AVE   	AAE   
# car                 	0.586 	0.437 	0.166 	0.133 	0.323 	0.208 
# truck               	0.364 	0.621 	0.210 	0.130 	0.265 	0.204 
# bus                 	0.400 	0.707 	0.188 	0.116 	0.527 	0.288 
# trailer             	0.226 	0.923 	0.244 	0.425 	0.201 	0.161 
# construction_vehicle	0.139 	0.792 	0.480 	1.483 	0.109 	0.346 
# pedestrian          	0.443 	0.662 	0.293 	0.753 	0.395 	0.237 
# motorcycle          	0.397 	0.534 	0.251 	0.690 	0.341 	0.199 
# bicycle             	0.379 	0.437 	0.257 	1.101 	0.142 	0.006 
# traffic_cone        	0.595 	0.377 	0.343 	nan   	nan   	nan   
# barrier             	0.576 	0.412 	0.277 	0.138 	nan   	nan   
#
# pts_bbox_NuScenes/car_AP_dist_0.5: 0.2692, pts_bbox_NuScenes/car_AP_dist_1.0: 0.5548, pts_bbox_NuScenes/car_AP_dist_2.0: 0.7243, pts_bbox_NuScenes/car_AP_dist_4.0: 0.7971, pts_bbox_NuScenes/car_trans_err: 0.4373, pts_bbox_NuScenes/car_scale_err: 0.166, pts_bbox_NuScenes/car_orient_err: 0.1328, pts_bbox_NuScenes/car_vel_err: 0.3234, pts_bbox_NuScenes/car_attr_err: 0.2075, pts_bbox_NuScenes/mATE: 0.5901, pts_bbox_NuScenes/mASE: 0.2708, pts_bbox_NuScenes/mAOE: 0.5521, pts_bbox_NuScenes/mAVE: 0.2879, pts_bbox_NuScenes/mAAE: 0.206, pts_bbox_NuScenes/truck_AP_dist_0.5: 0.082, pts_bbox_NuScenes/truck_AP_dist_1.0: 0.2803, pts_bbox_NuScenes/truck_AP_dist_2.0: 0.4945, pts_bbox_NuScenes/truck_AP_dist_4.0: 0.6008, pts_bbox_NuScenes/truck_trans_err: 0.6206, pts_bbox_NuScenes/truck_scale_err: 0.2099, pts_bbox_NuScenes/truck_orient_err: 0.1299, pts_bbox_NuScenes/truck_vel_err: 0.2646, pts_bbox_NuScenes/truck_attr_err: 0.2042, pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5: 0.0, pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0: 0.078, pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0: 0.1936, pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0: 0.2851, pts_bbox_NuScenes/construction_vehicle_trans_err: 0.7923, pts_bbox_NuScenes/construction_vehicle_scale_err: 0.4798, pts_bbox_NuScenes/construction_vehicle_orient_err: 1.483, pts_bbox_NuScenes/construction_vehicle_vel_err: 0.1088, pts_bbox_NuScenes/construction_vehicle_attr_err: 0.3456, pts_bbox_NuScenes/bus_AP_dist_0.5: 0.0485, pts_bbox_NuScenes/bus_AP_dist_1.0: 0.301, pts_bbox_NuScenes/bus_AP_dist_2.0: 0.558, pts_bbox_NuScenes/bus_AP_dist_4.0: 0.6937, pts_bbox_NuScenes/bus_trans_err: 0.7067, pts_bbox_NuScenes/bus_scale_err: 0.1882, pts_bbox_NuScenes/bus_orient_err: 0.1165, pts_bbox_NuScenes/bus_vel_err: 0.527, pts_bbox_NuScenes/bus_attr_err: 0.288, pts_bbox_NuScenes/trailer_AP_dist_0.5: 0.002, pts_bbox_NuScenes/trailer_AP_dist_1.0: 0.1014, pts_bbox_NuScenes/trailer_AP_dist_2.0: 0.3309, pts_bbox_NuScenes/trailer_AP_dist_4.0: 0.4695, pts_bbox_NuScenes/trailer_trans_err: 0.9228, pts_bbox_NuScenes/trailer_scale_err: 0.2437, pts_bbox_NuScenes/trailer_orient_err: 0.4248, pts_bbox_NuScenes/trailer_vel_err: 0.2013, pts_bbox_NuScenes/trailer_attr_err: 0.1612, pts_bbox_NuScenes/barrier_AP_dist_0.5: 0.3068, pts_bbox_NuScenes/barrier_AP_dist_1.0: 0.5927, pts_bbox_NuScenes/barrier_AP_dist_2.0: 0.6824, pts_bbox_NuScenes/barrier_AP_dist_4.0: 0.7235, pts_bbox_NuScenes/barrier_trans_err: 0.4121, pts_bbox_NuScenes/barrier_scale_err: 0.2767, pts_bbox_NuScenes/barrier_orient_err: 0.1383, pts_bbox_NuScenes/barrier_vel_err: nan, pts_bbox_NuScenes/barrier_attr_err: nan, pts_bbox_NuScenes/motorcycle_AP_dist_0.5: 0.1716, pts_bbox_NuScenes/motorcycle_AP_dist_1.0: 0.3768, pts_bbox_NuScenes/motorcycle_AP_dist_2.0: 0.4947, pts_bbox_NuScenes/motorcycle_AP_dist_4.0: 0.5453, pts_bbox_NuScenes/motorcycle_trans_err: 0.5335, pts_bbox_NuScenes/motorcycle_scale_err: 0.2509, pts_bbox_NuScenes/motorcycle_orient_err: 0.6899, pts_bbox_NuScenes/motorcycle_vel_err: 0.341, pts_bbox_NuScenes/motorcycle_attr_err: 0.1988, pts_bbox_NuScenes/bicycle_AP_dist_0.5: 0.2224, pts_bbox_NuScenes/bicycle_AP_dist_1.0: 0.3765, pts_bbox_NuScenes/bicycle_AP_dist_2.0: 0.4433, pts_bbox_NuScenes/bicycle_AP_dist_4.0: 0.4746, pts_bbox_NuScenes/bicycle_trans_err: 0.437, pts_bbox_NuScenes/bicycle_scale_err: 0.2572, pts_bbox_NuScenes/bicycle_orient_err: 1.1011, pts_bbox_NuScenes/bicycle_vel_err: 0.1417, pts_bbox_NuScenes/bicycle_attr_err: 0.0058, pts_bbox_NuScenes/pedestrian_AP_dist_0.5: 0.1327, pts_bbox_NuScenes/pedestrian_AP_dist_1.0: 0.3994, pts_bbox_NuScenes/pedestrian_AP_dist_2.0: 0.5756, pts_bbox_NuScenes/pedestrian_AP_dist_4.0: 0.666, pts_bbox_NuScenes/pedestrian_trans_err: 0.662, pts_bbox_NuScenes/pedestrian_scale_err: 0.2927, pts_bbox_NuScenes/pedestrian_orient_err: 0.7528, pts_bbox_NuScenes/pedestrian_vel_err: 0.3952, pts_bbox_NuScenes/pedestrian_attr_err: 0.2371, pts_bbox_NuScenes/traffic_cone_AP_dist_0.5: 0.3595, pts_bbox_NuScenes/traffic_cone_AP_dist_1.0: 0.5965, pts_bbox_NuScenes/traffic_cone_AP_dist_2.0: 0.6833, pts_bbox_NuScenes/traffic_cone_AP_dist_4.0: 0.742, pts_bbox_NuScenes/traffic_cone_trans_err: 0.3767, pts_bbox_NuScenes/traffic_cone_scale_err: 0.3432, pts_bbox_NuScenes/traffic_cone_orient_err: nan, pts_bbox_NuScenes/traffic_cone_vel_err: nan, pts_bbox_NuScenes/traffic_cone_attr_err: nan, pts_bbox_NuScenes/NDS: 0.5146845227563384, pts_bbox_NuScenes/mAP: 0.4107593228183424

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
plugin = True
plugin_dir = 'mmdet3d_plugin/'
data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
motion_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [1.0, 60.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

# load pretrained weights
load_from = 'work_dirs/hop_seabird_r50_256x704_stage1/epoch_4_ema.pth'
#resume_from="work_dirs/run_3/latest.pth"
samples_per_gpu = 8
history = 8
multi_adj_frame_id_cfg = (1, history+1, 1)

model = dict(
    type='HoPBEVDepth4D',
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    with_prev=True,
    with_hop=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False),
        downsample=16),
    history_decoder=dict(
        type='BiTemporalPredictor',
        in_channels=80,
        out_channels=256,
        embed_dims=160,
        num_adj=history-1,
        num_short=min(history,2),
        reduction=4,
        bev_h=128,
        bev_w=128,
        decoder_short=dict(
            type='TemporalDecoder',
            num_layers=2,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalCrossAttention',
                        embed_dims=160,
                        num_heads=5,
                        num_levels=1,
                        num_bev_queue=min(history,2),
                        dropout=0.0)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=160,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=512,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder_long=dict(
            type='TemporalDecoder',
            num_layers=2,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalCrossAttention',
                        embed_dims=160//4,
                        num_heads=2,
                        num_levels=1,
                        num_bev_queue=min(history,8),
                        dropout=0.0)
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=160//4,
                    feedforward_channels=128,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=128,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    aux_bbox_head=[dict(
        type='CenterHead',
        in_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3, norm_cfg=dict(type='SyncBN', requires_grad=True)),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0*0.5),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25*0.5),
        norm_bbox=True),],
    aux_train_cfg=[dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=[0.1, 0.1, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])],
    aux_test_cfg=[dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])],
    seg_head=[dict(
        type='MapHead',
        task_dict={
            'semantic_seg': 5,
        },
        in_channels=256,
        class_weights=[2.0, 2.0, 2.0, 2.0, 2.0],
        semantic_thresh=0.25,
        loss_type='dice',
        loss_weight=5.0,
    )],
    seabird=True,
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,

            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ])))

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True,
        add_adj_bbox=True,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        align_adj_bbox=True,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    # convert map labels
    dict(type='RasterizeMapVectors', map_grid_conf=motion_grid_conf, for_classes=True,seg_style='car_vehicle_barrier_pedestrian_cone'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'semantic_indices', 'semantic_map',
                                'gt_depth'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True, add_adj_bbox=True, file_client_args=file_client_args),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        align_adj_bbox=True,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    # convert map labels
    dict(type='RasterizeMapVectors', map_grid_conf= motion_grid_conf, for_classes=True, seg_style='car_vehicle_barrier_pedestrian_cone'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'semantic_indices', 'semantic_map'
                                         ])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

total_epochs = 24

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[24,])
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# logging
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_epoch=3,
    ),
]

# checkpointing
checkpoint_config = dict(interval=1)

# evaluation
# inherited from nus-3d.py
evaluation = dict(
    interval=1,
    pipeline=test_pipeline)

# fp16 = dict(loss_scale='dynamic')
