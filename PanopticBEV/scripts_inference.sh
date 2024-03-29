# KITTI-360 Val
CUDA_VISIBLE_DEVICES=0 python scripts/train_panoptic_bev.py  --project_root_dir=/home/abhinav/project/PanopticBEV/ --config=pbev_seabird_kitti360_val.ini --mode=test --resume output/pbev_seabird_kitti360_val/saved_models/model_19.pth --debug=True

# KITTI-360 Test
CUDA_VISIBLE_DEVICES=0 python scripts/train_panoptic_bev.py  --project_root_dir=/home/abhinav/project/PanopticBEV/ --config=pbev_seabird_kitti360_test.ini --mode=test --resume output/pbev_seabird_kitti360_test/saved_models/model_9.pth --debug=True

