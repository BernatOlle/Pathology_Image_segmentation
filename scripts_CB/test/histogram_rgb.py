import os
import json
import numpy as np
from PIL import Image

# --- Configuración de paths ---
input_dir = "/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level/train/NEP25/08_368_02/img/"
output_dir = "histogram"  # Carpeta donde se guardarán los resultados

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Variables para acumular estadísticas
sum_means = {'Red': 0, 'Green': 0, 'Blue': 0}
sum_variances = {'Red': 0, 'Green': 0, 'Blue': 0}
image_count = 0

# Procesar todas las imágenes en el directorio
for img_file in os.listdir(input_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Cargar imagen
            img_path = os.path.join(input_dir, img_file)
            img_array = np.array(Image.open(img_path))
            
            # Procesar cada canal
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                channel_data = img_array[:, :, i].ravel()
                sum_means[channel] += np.mean(channel_data)
                sum_variances[channel] += np.var(channel_data)
            
            image_count += 1
            
        except Exception as e:
            print(f"Error procesando {img_file}: {str(e)}")

# Calcular medias globales
global_stats = {}
for channel in ['Red', 'Green', 'Blue']:
    global_stats[channel] = {
        'mean': sum_means[channel] / image_count,
        'variance': sum_variances[channel] / image_count
    }

# Guardar en el formato específico solicitado
with open(os.path.join(output_dir, 'global_stats.json'), 'w') as f:
    json.dump(global_stats, f, indent=4)

# Mostrar resultados
print(f"Imágenes procesadas: {image_count}")
print("\nEstadísticas globales:")
print(json.dumps(global_stats, indent=4))