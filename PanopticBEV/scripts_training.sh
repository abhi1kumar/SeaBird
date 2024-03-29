# KITTI-360 Val Split
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 scripts/train_panoptic_bev.py --project_root_dir=/user/kumarab6/cvl/project/PanopticBEV/ --config=pbev_seabird_kitti360_val_stage1.ini  --mode=train
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 scripts/train_panoptic_bev.py --project_root_dir=/user/kumarab6/cvl/project/PanopticBEV/ --config=pbev_seabird_kitti360_val.ini --mode=train

# KITTI-360 Test Split
CUDA_VISIBLE_DEVICES=0,1,2,3         python -m torch.distributed.launch --nproc_per_node=4 scripts/train_panoptic_bev.py --project_root_dir=/user/kumarab6/cvl/project/PanopticBEV/ --config=pbev_seabird_kitti360_test.ini --mode=train

# Single Core
# CUDA_VISIBLE_DEVICES=0               python -u                                             scripts/train_panoptic_bev.py --project_root_dir=/user/kumarab6/cvl/project/PanopticBEV/  --config=run_9.ini  --mode=train --resume output/run_9/saved_models/model_1.pth --debug=True
