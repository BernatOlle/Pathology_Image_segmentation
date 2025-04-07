#!/bin/bash
python inference_wsi_level_kpis.py \
--input /mnt/work/datasets/BKidney/KPIS/KPIs24\ Testing\ Data//Task2_WSI_level/NEP25/18-579_wsi.tiff \
--config segformer/segformer_mit-b5_kpis_isbi_768.py \
--ckpt segformer/segformer_mit_b5_kpis_768_best_mDice.pth \
--patch_size 2048 --stride 1024