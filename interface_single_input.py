import cv2
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope
import os
import numpy as np

# init the default transform to mmseg
init_default_scope('mmseg')

# define test_pipeline
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='PackSegInputs'),
]

# example: load SegFormer model
config_path = '/home/usuaris/imatge/constanza.elfarkh/Pathology_Image_segmentation/segformer/segformer_mit-b5_kpis_isbi_768.py'
ckpt_path = '/home/usuaris/imatge/constanza.elfarkh/Pathology_Image_segmentation/segformer/segformer_mit_b5_kpis_768_best_mDice.pth'
model = init_model(config_path, ckpt_path)

# assign test_pipeline
model.cfg.test_pipeline = test_pipeline

# inference
img_data = cv2.imread('/home/usuaris/imatge/constanza.elfarkh/Pathology_Image_segmentation/output_CROC_level0_imgS17_TINTEMORADO/slide-2023-02-18T08-17-59-R3-S17/slide-2023-02-18T08-17-59-R3-S17_patch0018_x30720_y108544.png', -1)
pred_res = inference_model(model, img_data)

# get the predicted mask
raw_logits = pred_res.seg_logits.data
_, pred_mask = raw_logits.max(axis=0, keepdims=True)
pred_mask = pred_mask.cpu().numpy()[0]

# Crear la carpeta pred_masks si no existe
output_dir = 'pred_masks'
os.makedirs(output_dir, exist_ok=True)

# Guardar la máscara como imagen PNG
mask_filename = os.path.join(output_dir, 'predicted_mask.png')
cv2.imwrite(mask_filename, pred_mask.astype(np.uint8) * 255)  # Escalar a 0-255

print(f"Máscara guardada en: {mask_filename}")  