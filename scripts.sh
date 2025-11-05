CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN \
accelerate launch --multi_gpu main.py   --data_root /data/mingxing/IMNET100K/   --out_dir runs/ldm_imnet256   --amp --per_device_batch 21 --unet_epochs 600


# CUDA_VISIBLE_DEVICES=4,5,6,7
# accelerate launch --multi_gpu main.py \
#   --data_root /banana/ethan/MIA_data/IMAGENET1K/data_ori \
#   --out_dir runs/ldm_imnet256 \
#   --amp

# python attack.py --data_root /data/mingxing/IMNET100K/ --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k --amp

# python attack.py --data_root /data/mingxing/IMNET100K/ --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k

python cal_pullback.py --data_root /data/mingxing/IMNET100K/ --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k --out_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_pullback.npz


python attack_by_group_advance.py \
  --data_root /data/mingxing/IMNET100K/ \
  --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k \
  --pullback_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_pullback.npz


python cal_per_dim_contri.py \
  --data_root /data/mingxing/IMNET100K/ \
  --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k \
  --pullback_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_pullback.npz \
  --out_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_per_dim.npz

python attack_per_dim.py \
  --data_root /data/mingxing/IMNET100K/data \
  --out_dir /data/mingxing/IMNET100K/runs/ldm_imnet256_10k \
  --pullback_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_pullback.npz \
  --perdim_npz /data/mingxing/IMNET100K/runs/ldm_imnet256_10k/imnetv1_10k_per_dim.npz
