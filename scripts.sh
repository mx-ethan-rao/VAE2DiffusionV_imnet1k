CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN \
accelerate launch --multi_gpu main.py   --data_root /data/mingxing/IMNET100K/   --out_dir runs/ldm_imnet256   --amp --per_device_batch 21 --unet_epochs 600


# CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --multi_gpu main.py \
  --data_root /banana/ethan/MIA_data/IMAGENET1K/data_ori \
  --out_dir runs/ldm_imnet256 \
  --amp

python attack.py --data_root /data/mingxing/IMNET100K/ --out_dir /home/ethanrao/MIA_LDM/ldm4imagenet/runs/ldm_imnet256 --amp

python attack.py --data_root /data/mingxing/IMNET100K/ --out_dir /home/ethanrao/MIA_LDM/ldm4imagenet/runs/ldm_imnet256
python attack_by_group_advance.py --data_root /data/mingxing/IMNET100K/ --out_dir /home/ethanrao/MIA_LDM/ldm4imagenet/runs/ldm_imnet256 --pullback_npz /home/ethanrao/MIA_LDM/ldm4imagenet/runs/ldm_imnet256/imnetv1_3k_pullback.npz
