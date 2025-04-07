import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def match_histograms(source_img, reference_stats):
    """
    Adapta el histograma de la imagen fuente para que coincida con las estadísticas de referencia
    """
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):  # Para cada canal RGB
        # Datos del canal fuente
        source_data = source_img[:, :, channel].astype(np.float32)
        source_mean = np.mean(source_data)
        source_std = np.std(source_data)
        
        # Estadísticas de referencia
        ref_mean = reference_stats[['Red', 'Green', 'Blue'][channel]]['mean']
        ref_std = np.sqrt(reference_stats[['Red', 'Green', 'Blue'][channel]]['variance'])
        
        # Evit ar división por cero
        if source_std == 0:
            source_std = 1
        
        # Aplicar transformación lineal para igualar estadísticas
        matched_data = (source_data - source_mean) * (ref_std / source_std) + ref_mean
        
        # Asegurar valores dentro del rango [0, 255]
        matched_data = np.clip(matched_data, 0, 255)
        matched_img[:, :, channel] = matched_data.astype(np.uint8)
    
    return matched_img

# --- Configuración de paths ---
reference_stats_path = "/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/scripts_CB/test/histogram/global_stats.json"  # JSON con estadísticas de referencia
input_image_path = "/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/result/slide-2023-02-18T08-17-59-R3-S17/paches/no_tinted/slide-2023-02-18T08-17-59-R3-S17_patch0000_x20480_y104448.png"  # Nueva imagen a transformar
output_dir = "histogram_matched"  # Carpeta para resultados

# Crear carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# --- Cargar estadísticas de referencia ---
with open(reference_stats_path, 'r') as f:
    reference_stats = json.load(f)

# --- Procesar nueva imagen ---
source_img = np.array(Image.open(input_image_path))

# Aplicar transferencia de histograma
matched_img = match_histograms(source_img, reference_stats)

# --- Guardar resultados ---
# Obtener nombre base del archivo
image_name = os.path.splitext(os.path.basename(input_image_path))[0]

# Guardar imagen transformada
output_path = os.path.join(output_dir, f"{image_name}_matched.jpg")
Image.fromarray(matched_img).save(output_path)

# --- Calcular y mostrar histogramas ---
plt.figure(figsize=(15, 5))

# Histograma original
plt.subplot(1, 2, 1)
plt.title("Original Image Histogram")
for i, color in enumerate(['red', 'green', 'blue']):
    hist, _ = np.histogram(source_img[:, :, i].ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color=color, label=['Red', 'Green', 'Blue'][i])
plt.legend()
plt.grid(alpha=0.3)

# Histograma transformado
plt.subplot(1, 2, 2)
plt.title("Matched Histogram")
for i, color in enumerate(['red', 'green', 'blue']):
    hist, _ = np.histogram(matched_img[:, :, i].ravel(), bins=256, range=(0, 256))
    plt.plot(hist, color=color, label=['Red', 'Green', 'Blue'][i])
plt.legend()
plt.grid(alpha=0.3)

# Guardar gráfico de histogramas
histogram_plot_path = os.path.join(output_dir, f"{image_name}_histogram_comparison.png")
plt.savefig(histogram_plot_path, bbox_inches='tight')
plt.close()

print(f"✅ Imagen transformada guardada en: {output_path}")
print(f"✅ Comparación de histogramas guardada en: {histogram_plot_path}")