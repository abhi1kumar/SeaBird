# =============================================================================
# nuScenes Val Models
# =============================================================================
# HoP+SeaBird on 256x704
CUDA_VISIBLE_DEVICES=0,1,2,3       /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_r50_256x704_stage1.py    --work-dir=work_dirs/hop_seabird_r50_256x704_stage1    --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3       /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_r50_256x704.py           --work-dir=work_dirs/hop_seabird_r50_256x704           --seed 0 --deterministic

# HoP+SeaBird on 512x1408
CUDA_VISIBLE_DEVICES=0,1,2,3       /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_r101_512x1408_stage1.py  --work-dir=work_dirs/hop_seabird_r101_512x1408_stage1  --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,3       /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_r101_512x1408.py         --work-dir=work_dirs/hop_seabird_r101_512x1408         --seed 0 --deterministic

# HoP+SeaBird on 640x1600
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_vov99_640x1600_stage1.py --work-dir=work_dirs/hop_seabird_vov99_640x1600_stage1 --seed 0 --deterministic
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7 /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_vov99_640x1600.py        --work-dir=work_dirs/hop_seabird_vov99_640x1600        --seed 0 --deterministic

# =============================================================================
# nuScenes Test Models
# =============================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3       /mnt/home/kumarab6/anaconda3/envs/hop2/bin/python -u tools/train.py configs/hop_bevdet/hop_seabird_r101_512x1408_test.py         --work-dir=work_dirs/hop_seabird_r101_512x1408_test --seed 0 --deterministic
