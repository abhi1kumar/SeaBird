# [SeaBird: Segmentation in Bird's View with Dice Loss Improves Monocular 3D Detection of Large Objects](https://arxiv.org/pdf/2403.20318.pdf)

[Abhinav Kumar](https://sites.google.com/view/abhinavkumar)<sup>1</sup>, 
[Yuliang Guo](https://yuliangguo.github.io)<sup>2</sup>, 
[Xinyu Huang](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en)<sup>2</sup>, 
[Liu Ren](https://www.liu-ren.com)<sup>2</sup>, 
[Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)<sup>1</sup> <br>
<sup>1</sup>Michigan State University, <sup>2</sup>Bosch Research North America, Bosch Center for AI

in [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/)

<img src="/PanopticBEV/images/Seabird_pipeline.png" width="1000" >

> Monocular 3D detectors achieve remarkable performance on cars and smaller objects. However, their performance drops on larger objects, leading to fatal accidents. Some attribute the failures to training data scarcity or their receptive field requirements of large objects. In this paper, we highlight this understudied problem of generalization to large objects. We find that modern frontal detectors struggle to generalize to large objects even on nearly balanced datasets. We argue that the cause of failure is the sensitivity of depth regression losses to noise of larger objects. To bridge this gap, we comprehensively investigate regression and dice losses, examining their robustness under varying error levels and object sizes. We mathematically prove that the dice loss leads to superior noise-robustness and model convergence for large objects compared to regression losses for a simplified case. Leveraging our theoretical insights, we propose SeaBird (Segmentation in Bird's View) as the first step towards generalizing to large objects. SeaBird effectively integrates BEV segmentation on foreground objects for 3D detection, with the segmentation head trained with the dice loss. SeaBird achieves SoTA results on the KITTI-360 leaderboard and improves existing detectors on the nuScenes leaderboard, particularly for large objects.

Much of the codebase is based on [HoP](https://github.com/Sense-X/HoP).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Citation

If you find our work useful in your research, please consider starring the repo and citing:

```Bibtex
@inproceedings{kumar2024seabird,
   title={{SeaBird: Segmentation in Bird's View with Dice Loss Improves Monocular $3$D Detection of Large Objects}},
   author={Kumar, Abhinav and Guo, Yuliang and Huang, Xinyu and Ren, Liu and Liu, Xiaoming},
   booktitle={CVPR},
   year={2024}
}
```

## Setup

### Environment
  We train our models under the following environment: 

  ```bash
  module CUDA/11.0.2 GCCcore/9.1.0  GCC/9.1.0-2.32
  source cuda_11.1_env
  conda create -n hop2 python=3.8 -y
  conda install -c anaconda ipython -y
  pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
  pip install openmim
  mim install mmcv-full==1.5.2
  mim install mmengine
  pip install mmdet==2.24.0
  pip install mmsegmentation==0.30.0
  git clone git@github.com:abhi1kumar/SeaBird.git
  cd HoP
  pip install -e .
  pip install numba numpy==1.23.5 timm einops yapf==0.40.1
  ```

   The source code of MMDetection3D has been included in this repo.

### Data Preparation

  Follow the steps to prepare nuScenes Dataset introduced in [nuscenes_det.md](https://github.com/HuangJunJie2017/BEVDet/blob/dev2.1/docs/en/datasets/nuscenes_det.md). 
  Then, create the pickles by running:

  ```bash
  python tools/create_data_bevdet.py
  python tools/create_data_bevdet.py --split test
  ```

  This should create `bevdetv2-nuscenes_infos_train.pkl`, `bevdetv2-nuscenes_infos_val.pkl` and `bevdetv2-nuscenes_infos_test.pkl` pickles inside the `data/nuscenes` directory. 

### Pretrained Checkpoints

   Make `pretrain` folder in the `SeaBird/HoP` directory:
   
   ```bash
   mkdir pretrain
   ```
  Download the pretrained [V2-99 checkpoint](https://drive.google.com/file/d/1Zlhum7DD7KYaQjpKaTGtIgcEFlxyVkvU/view?usp=sharing) trained on DDAD15M dataset and place inside `pretrain` folder.

   The files should be arranged as follows:
   ```bash
   SeaBird/HoP
   ├── data
   │      └── nuscenes
   │             ├── bevdetv2-nuscenes_infos_test.pkl
   │             ├── bevdetv2-nuscenes_infos_train.pkl
   │             ├── bevdetv2-nuscenes_infos_val.pkl
   │             ├── maps
   │             ├── samples
   │             ├── sweeps
   │             ├── v1.0-mini
   │             ├── v1.0-test
   │             └── v1.0-trainval
   │
   ├── pretrain
   │      └── dd3d_det_final.pth
```

## Training

Train the model:

```bash
chmod +x scripts_training.sh
bash scripts_training.sh
```

## Testing

### Model Zoo

**nuScenes Val Results**

|          Model           | Resolution | Backbone     | Pretrain | APLrg |  mAP   |  NDS   |                      Ckpt/Log/Pred                       |
| :----------------------: | :--------: | :----------: | :------: | :---: | :----: | :----: | :------------------------------------------------------: |
| [HoP_BEVDet4D_256](configs/hop_bevdet/hop_bevdet4d-r50-depth.py)    | 256x704  |  ResNet50   | ImageNet-1K | 0.274 | 0.399 | 0.509 | [ckpt](https://github.com/Sense-X/HoP/releases/download/Release/HoP_BEVDet_ep24_ema.pth) / [log](https://github.com/Sense-X/HoP/releases/download/Release/HoP_BEVDet.log) |
| [HoP+SeaBird_256 Stage1](configs/hop_bevdet/hop_seabird_r50_256x704_stage1.py)    | 256x704  |  ResNet50   | ImageNet-1K | - | - | -| [gdrive](https://drive.google.com/file/d/1zQOO2A4Twno6C1nC53RCoazDVuZ_9W0e/view?usp=sharing) | 
| [HoP+SeaBird_256](configs/hop_bevdet/hop_seabird_r50_256x704.py)    | 256x704  |  ResNet50   | ImageNet-1K | 0.282 | 0.411 | 0.515 | [gdrive](https://drive.google.com/file/d/1dmVQW8yDE423mm6AKBM2o1_3F3V59dXQ/view?usp=sharing) |   
| [HoP+SeaBird_512 Stage1](configs/hop_bevdet/hop_seabird_r101_512x1408_stage1.py)   | 512x1408 |  ResNet101   | ImageNet-1K | - | - | - | [gdrive](https://drive.google.com/file/d/1D44imXsFSYg9WE-kdnE-SH-G_gjXvxlw/view?usp=sharing) |
| [HoP+SeaBird_512](configs/hop_bevdet/hop_seabird_r101_512x1408.py)   | 512x1408 |  ResNet101   | ImageNet-1K | 0.329 | 0.462 | 0.547 | [gdrive](https://drive.google.com/file/d/1pgMzLGjXh5A_P3XR7CmQju5qXenTAXAJ/view?usp=sharing) |
| [HoP+SeaBird_640 Stage1](configs/hop_bevdet/hop_seabird_vov99_640x1600_stage1.py)   | 640x1600 |  V2-99   | DDAD15M | - | - | - | [gdrive](https://drive.google.com/file/d/1cbVkituogo_e5ILMrC8Z8NdemrfQL2uV/view?usp=sharing) |
| [HoP+SeaBird_640](configs/hop_bevdet/hop_seabird_vov99_640x1600.py)   | 640x1600 |  V2-99   | DDAD15M | 0.403 | 0.527 | 0.602 | [gdrive](https://drive.google.com/file/d/1dz1w0DQrjgw1xm6u6Kp4csNThYkYPFhu/view?usp=sharing) |

**nuScenes Test Results**

|          Model           | Resolution | Backbone     | Pretrain | APLrg |  mAP   |  NDS   |                      Ckpt/Log/Pred                       |
| :----------------------: | :--------: | :----------: | :------: | :---: | :----: | :----: | :------------------------------------------------------: |
| [HoP+SeaBird_512 Test](configs/hop_bevdet/hop_seabird_r101_512x1408_test.py)   | 512x1408 |  ResNet101   | ImageNet-1K | 0.366 | 0.486 | 0.570 | [gdrive](https://drive.google.com/file/d/1Y39kVTdw0OXN45u6UedgXfYhePFZ-0Xt/view?usp=sharing) |
| [HoP+SeaBird_640 Val](configs/hop_bevdet/hop_seabird_vov99_640x1600.py)   | 640x1600 |  V2-99   | DDAD15M | 0.384 | 0.511 | 0.597 | [gdrive](https://drive.google.com/file/d/1dz1w0DQrjgw1xm6u6Kp4csNThYkYPFhu/view?usp=sharing) |

Please submit the test JSON to the nuScenes evaluation server to get these results.

### Testing Pre-trained Models

Make `work_dirs` folder in the `SeaBird/HoP` directory:

```bash
mkdir work_dirs
```

Place models in the `work_dirs` folder as follows:

```bash
SeaBird/HoP
├── work_dirs
│      ├── hop_seabird_r50_256x704
│      │       └── epoch_24_ema.pth
│      │
│      ├── hop_seabird_r101_512x1408
│      │       └── epoch_24_ema.pth
│      │
│      ├── hop_seabird_vov99_640x1600
│      │       └── epoch_24_ema.pth
│      │
│      └── hop_seabird_r101_512x1408_test
│              └── epoch_4_ema.pth
```

To test, execute the following command:
```bash
chmod +x scripts_inference.sh
bash scripts_inference.sh 
```

To get the AP_Lrg, AP_car and AP_small numbers as we report in the paper, use the `tools/parse_nuscenes_log.py` function with the `--str` argument and the log data. As an example:
```bash
python tools/parse_nuscenes_log.py --str "pts_bbox_NuScenes/car_AP_dist_0.5: 0.2692, pts_bbox_NuScenes/car_AP_dist_1.0: 0.5548, pts_bbox_NuScenes/car_AP_dist_2.0: 0.7243, pts_bbox_NuScenes/car_AP_dist_4.0: 0.7971, pts_bbox_NuScenes/car_trans_err: 0.4373, pts_bbox_NuScenes/car_scale_err: 0.166, pts_bbox_NuScenes/car_orient_err: 0.1328, pts_bbox_NuScenes/car_vel_err: 0.3234, pts_bbox_NuScenes/car_attr_err: 0.2075, pts_bbox_NuScenes/mATE: 0.5901, pts_bbox_NuScenes/mASE: 0.2708, pts_bbox_NuScenes/mAOE: 0.5521, pts_bbox_NuScenes/mAVE: 0.2879, pts_bbox_NuScenes/mAAE: 0.206, pts_bbox_NuScenes/truck_AP_dist_0.5: 0.082, pts_bbox_NuScenes/truck_AP_dist_1.0: 0.2803, pts_bbox_NuScenes/truck_AP_dist_2.0: 0.4945, pts_bbox_NuScenes/truck_AP_dist_4.0: 0.6008, pts_bbox_NuScenes/truck_trans_err: 0.6206, pts_bbox_NuScenes/truck_scale_err: 0.2099, pts_bbox_NuScenes/truck_orient_err: 0.1299, pts_bbox_NuScenes/truck_vel_err: 0.2646, pts_bbox_NuScenes/truck_attr_err: 0.2042, pts_bbox_NuScenes/construction_vehicle_AP_dist_0.5: 0.0, pts_bbox_NuScenes/construction_vehicle_AP_dist_1.0: 0.078, pts_bbox_NuScenes/construction_vehicle_AP_dist_2.0: 0.1936, pts_bbox_NuScenes/construction_vehicle_AP_dist_4.0: 0.2851, pts_bbox_NuScenes/construction_vehicle_trans_err: 0.7923, pts_bbox_NuScenes/construction_vehicle_scale_err: 0.4798, pts_bbox_NuScenes/construction_vehicle_orient_err: 1.483, pts_bbox_NuScenes/construction_vehicle_vel_err: 0.1088, pts_bbox_NuScenes/construction_vehicle_attr_err: 0.3456, pts_bbox_NuScenes/bus_AP_dist_0.5: 0.0485, pts_bbox_NuScenes/bus_AP_dist_1.0: 0.301, pts_bbox_NuScenes/bus_AP_dist_2.0: 0.558, pts_bbox_NuScenes/bus_AP_dist_4.0: 0.6937, pts_bbox_NuScenes/bus_trans_err: 0.7067, pts_bbox_NuScenes/bus_scale_err: 0.1882, pts_bbox_NuScenes/bus_orient_err: 0.1165, pts_bbox_NuScenes/bus_vel_err: 0.527, pts_bbox_NuScenes/bus_attr_err: 0.288, pts_bbox_NuScenes/trailer_AP_dist_0.5: 0.002, pts_bbox_NuScenes/trailer_AP_dist_1.0: 0.1014, pts_bbox_NuScenes/trailer_AP_dist_2.0: 0.3309, pts_bbox_NuScenes/trailer_AP_dist_4.0: 0.4695, pts_bbox_NuScenes/trailer_trans_err: 0.9228, pts_bbox_NuScenes/trailer_scale_err: 0.2437, pts_bbox_NuScenes/trailer_orient_err: 0.4248, pts_bbox_NuScenes/trailer_vel_err: 0.2013, pts_bbox_NuScenes/trailer_attr_err: 0.1612, pts_bbox_NuScenes/barrier_AP_dist_0.5: 0.3068, pts_bbox_NuScenes/barrier_AP_dist_1.0: 0.5927, pts_bbox_NuScenes/barrier_AP_dist_2.0: 0.6824, pts_bbox_NuScenes/barrier_AP_dist_4.0: 0.7235, pts_bbox_NuScenes/barrier_trans_err: 0.4121, pts_bbox_NuScenes/barrier_scale_err: 0.2767, pts_bbox_NuScenes/barrier_orient_err: 0.1383, pts_bbox_NuScenes/barrier_vel_err: nan, pts_bbox_NuScenes/barrier_attr_err: nan, pts_bbox_NuScenes/motorcycle_AP_dist_0.5: 0.1716, pts_bbox_NuScenes/motorcycle_AP_dist_1.0: 0.3768, pts_bbox_NuScenes/motorcycle_AP_dist_2.0: 0.4947, pts_bbox_NuScenes/motorcycle_AP_dist_4.0: 0.5453, pts_bbox_NuScenes/motorcycle_trans_err: 0.5335, pts_bbox_NuScenes/motorcycle_scale_err: 0.2509, pts_bbox_NuScenes/motorcycle_orient_err: 0.6899, pts_bbox_NuScenes/motorcycle_vel_err: 0.341, pts_bbox_NuScenes/motorcycle_attr_err: 0.1988, pts_bbox_NuScenes/bicycle_AP_dist_0.5: 0.2224, pts_bbox_NuScenes/bicycle_AP_dist_1.0: 0.3765, pts_bbox_NuScenes/bicycle_AP_dist_2.0: 0.4433, pts_bbox_NuScenes/bicycle_AP_dist_4.0: 0.4746, pts_bbox_NuScenes/bicycle_trans_err: 0.437, pts_bbox_NuScenes/bicycle_scale_err: 0.2572, pts_bbox_NuScenes/bicycle_orient_err: 1.1011, pts_bbox_NuScenes/bicycle_vel_err: 0.1417, pts_bbox_NuScenes/bicycle_attr_err: 0.0058, pts_bbox_NuScenes/pedestrian_AP_dist_0.5: 0.1327, pts_bbox_NuScenes/pedestrian_AP_dist_1.0: 0.3994, pts_bbox_NuScenes/pedestrian_AP_dist_2.0: 0.5756, pts_bbox_NuScenes/pedestrian_AP_dist_4.0: 0.666, pts_bbox_NuScenes/pedestrian_trans_err: 0.662, pts_bbox_NuScenes/pedestrian_scale_err: 0.2927, pts_bbox_NuScenes/pedestrian_orient_err: 0.7528, pts_bbox_NuScenes/pedestrian_vel_err: 0.3952, pts_bbox_NuScenes/pedestrian_attr_err: 0.2371, pts_bbox_NuScenes/traffic_cone_AP_dist_0.5: 0.3595, pts_bbox_NuScenes/traffic_cone_AP_dist_1.0: 0.5965, pts_bbox_NuScenes/traffic_cone_AP_dist_2.0: 0.6833, pts_bbox_NuScenes/traffic_cone_AP_dist_4.0: 0.742, pts_bbox_NuScenes/traffic_cone_trans_err: 0.3767, pts_bbox_NuScenes/traffic_cone_scale_err: 0.3432, pts_bbox_NuScenes/traffic_cone_orient_err: nan, pts_bbox_NuScenes/traffic_cone_vel_err: nan, pts_bbox_NuScenes/traffic_cone_attr_err: nan, pts_bbox_NuScenes/NDS: 0.5146845227563384, pts_bbox_NuScenes/mAP: 0.4107593228183424"
```

## Acknowledgements
We thank the authors of the following awesome codebases:
- [HoP](https://github.com/Sense-X/HoP)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

Please also consider citing them.

## Contributions
We welcome contributions to the SeaBird repo. Feel free to raise a pull request.

## License
SeaBird and HoP code are under the [MIT license](https://opensource.org/license/mit).
