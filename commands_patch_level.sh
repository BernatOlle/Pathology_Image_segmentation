#!/bin/bash
python inference_patchlevel.py \
--input /home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/output_CROC_2/slide-2023-02-18T08-17-59-R3-S17 \
--config segformer/segformer_mit-b5_kpis_isbi_768.py  \
--ckpt segformer/segformer_mit_b5_kpis_768_best_mDice.pth \
--stitch \
--output output_data/

