import cv2
import numpy as np
from mmseg.apis import init_model
from mmseg.datasets import KPIsDataset
from mmengine.dataset import build_dataset  # Importar desde mmengine

# Carga configuración y modelo
config_path = '/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit-b5_kpis_isbi_768.py'
ckpt_path = '/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/segformer/segformer_mit_b5_kpis_768_best_mDice.pth'
model = init_model(config_path, ckpt_path)

# Cargar configuración
from mmengine import Config
cfg = Config.fromfile(config_path)

# Construir dataset usando la función build_dataset de mmengine
dataset = build_dataset(cfg.train_dataloader.dataset)

# Alternativa si build_dataset no funciona:
# Si es un ConcatDataset, tomar solo la primera configuración de dataset
# dataset = KPIsDataset(**cfg.train_dataloader.dataset.datasets[0])

# Verifica una muestra aleatoria
sample = dataset[0]
mask = sample['data_samples'].gt_sem_seg.data[0].numpy()
print("Valores únicos en máscara:", np.unique(mask))
print("Porcentaje de píxeles positivos:", np.mean(mask > 0))

