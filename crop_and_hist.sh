#!/bin/bash
python crop_and_hist.py \
--input_dir /ruta/a/tus/mrxs \
--compute_reference --training_patches_dir "/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level/train" \
--histogram_match_method distribution