#!/bin/bash

python mrxs_segmentation.py \
  --input_dir  /mnt/work/datasets/BKidney/CROC/slide-2023-02-18T08-17-59-R3-S17.mrxs \
  --patch_size 2048 \
  --stride 2048 \
  --level 0 \
  --config_path /home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit-b5_kpis_isbi_768.py \
  --ckpt_path /home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit_b5_kpis_768_best_mDice.pth