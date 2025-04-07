#!/bin/bash
python inference_patchlevel.py \
--input /home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/result/slide-2023-02-18T08-17-59-R3-S17 \
--config Mask2Former/mask2former_swin-b_kpis_isbi_768.py  \
--ckpt Mask2Former/mask2former_swin_b_kpis_768_best_mDice.pth \
--stitch 