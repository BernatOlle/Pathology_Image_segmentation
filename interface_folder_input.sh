#!/bin/bash
# Archivo: run.sh
# Asegúrate de darle permisos de ejecución: chmod +x run.sh

# Parámetros
INPUT_DIR="/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/result/slide-2023-02-18T08-17-59-R3-S17"
CONFIG="/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit-b5_kpis_isbi_768.py"
CKPT="/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit_b5_kpis_768_best_mDice.pth"
MASK_DIR=""  # Si tienes un directorio con máscaras de referencia, especifícalo; de lo contrario, déjalo vacío.
SCALE_FACTOR=4

# Ejecutar el script Python
python3 interface_folder_input.py --input_dir "$INPUT_DIR" --config "$CONFIG" --ckpt "$CKPT" --mask_dir "$MASK_DIR" --scale_factor "$SCALE_FACTOR"
