import cv2
import numpy as np

# Carga una muestra aleatoria
img = cv2.imread('/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level/train/normal/normal_F4/img/normal_F4_5_5120_0_img.jpg')  # Valores deberían ser 0-255
mask = cv2.imread('/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level/train/normal/normal_F4/mask/normal_F4_5_5120_0_mask.jpg', cv2.IMREAD_GRAYSCALE)  # Valores deberían ser 0/255 o 0/1

print(f"Imagen - Min: {img.min()}, Max: {img.max()}")
print(f"Máscara - Valores únicos: {np.unique(mask)}")