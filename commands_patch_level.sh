#!/bin/bash
python inference_patch_level_2.py \
--input /mnt/work/datasets/BKidney/KPIS/KPIs24\ Testing\ Data/Task1_patch_level/test/DN/11-363 \
--config segformer/segformer_mit-b5_kpis_isbi_768.py  \
--ckpt segformer/segformer_mit_b5_kpis_768_best_mDice.pth \
--img_size 2048 --stitch
