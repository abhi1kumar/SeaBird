# =============================================================================
# nuScenes Val Models
# =============================================================================
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/hop_bevdet/hop_seabird_r50_256x704.py    work_dirs/hop_seabird_r50_256x704/epoch_24_ema.pth    --eval bbox --deterministic
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/hop_bevdet/hop_seabird_r101_512x1408.py  work_dirs/hop_seabird_r101_512x1408/epoch_24_ema.pth  --eval bbox --deterministic
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/hop_bevdet/hop_seabird_vov99_640x1600.py work_dirs/hop_seabird_vov99_640x1600/epoch_24_ema.pth --eval bbox --deterministic

# =============================================================================
# nuScenes Test Models
# =============================================================================
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/hop_bevdet/hop_seabird_r101_512x1408_test.py  work_dirs/hop_seabird_r101_512x1408_test/epoch_24_ema.pth  --eval bbox --deterministic

# For test, change test_data_config ann_file to data_root + 'bevdetv2-nuscenes_infos_test.pkl'
CUDA_VISIBLE_DEVICES=0 python -u tools/test.py configs/hop_bevdet/hop_seabird_vov99_640x1600.py work_dirs/hop_seabird_vov99_640x1600/epoch_24_ema.pth --eval bbox --deterministic
