import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image
import json
import cv2
from skimage import exposure
from scipy import ndimage
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description='Recortar imágenes médicas MRXS en parches con normalización avanzada de histograma'
    )
    parser.add_argument('--input_dir', type=str, required=True, help='Archivo MRXS o directorio con imágenes MRXS')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024, help='Stride (paso) para el recorte (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar (default: 0)')
    parser.add_argument('--stats_json', type=str, default="/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/scripts_CB/test/histogram/global_stats.json", help='Ruta al JSON con estadísticas de referencia')
    parser.add_argument('--extreme_threshold', type=float, default=0.95, help='Umbral para filtrar imágenes con píxeles extremos (0-1)')
    parser.add_argument('--min_variance', type=float, default=10.0, help='Varianza mínima para considerar la imagen válida')
    parser.add_argument('--norm_method', type=str, default='adaptive', choices=['basic', 'adaptive', 'clahe', 'reinhard', 'hybrid'], help='Método de normalización a usar')
    parser.add_argument('--preserve_ratio', type=float, default=0.2, help='Ratio de preservación para estructuras importantes (0-1)')
    parser.add_argument('--denoise', action='store_true', help='Aplicar reducción de ruido antes de normalizar')
    parser.add_argument('--visualize', action='store_true', help='Generar visualizaciones de histogramas')
    parser.add_argument('--tissue_threshold', type=float, default=0.8, help='Umbral para detección de tejido (0-1)')
    parser.add_argument('--skip_background', action='store_true', help='Ignorar fondo en la normalización')
    parser.add_argument('--adaptive_strength', type=float, default=0.7, help='Fuerza de la adaptación (0-1)')
    return parser.parse_args()

def detect_tissue_mask(img, threshold=0.8):
    """
    Detecta una máscara de tejido en la imagen, eliminando el fondo.
    Devuelve una máscara binaria donde 1=tejido, 0=fondo.
    """
    # Convertir a escala de grises
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Binarizar usando el método de Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operaciones morfológicas para mejorar la máscara
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Convertir a formato binario 0-1
    mask = mask.astype(bool).astype(np.uint8)
    
    # Opcional: eliminar componentes pequeños
    if threshold < 1.0:
        # Etiquetado de componentes conectados
        labeled, num = ndimage.label(mask)
        sizes = np.bincount(labeled.ravel())
        if len(sizes) > 1:  # Si hay al menos un componente además del fondo
            sizes[0] = 0  # Ignorar el fondo
            mask_sizes = sizes > (threshold * np.max(sizes))
            mask = mask_sizes[labeled]
    
    return mask

def denoise_image(img, strength=10):
    """
    Aplica reducción de ruido preservando bordes.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)

def visualize_histograms(original, normalized, output_path):
    """
    Genera visualizaciones de histogramas antes y después de la normalización.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Imágenes originales y normalizadas
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(normalized)
    axes[1, 0].set_title('Normalizada')
    axes[1, 0].axis('off')
    
    # Histogramas para cada canal
    colors = ('r', 'g', 'b')
    channel_names = ('Rojo', 'Verde', 'Azul')
    
    for i, color in enumerate(colors):
        # Histograma original
        hist = cv2.calcHist([original], [i], None, [256], [0, 256])
        axes[0, i+1].plot(hist, color=color)
        axes[0, i+1].set_title(f'Histograma {channel_names[i]} Original')
        axes[0, i+1].set_xlim([0, 256])
        
        # Histograma normalizado
        hist = cv2.calcHist([normalized], [i], None, [256], [0, 256])
        axes[1, i+1].plot(hist, color=color)
        axes[1, i+1].set_title(f'Histograma {channel_names[i]} Normalizado')
        axes[1, i+1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def match_histograms_basic(source_img, reference_stats, alpha=1.0):
    """
    Versión mejorada del método original de adaptación de histogramas.
    """
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):
        # Datos del canal fuente
        source_data = source_img[:, :, channel].astype(np.float32)
        
        # Normalizar a [0, 1]
        source_normalized = source_data / 255.0
        
        # Aplicar función sigmoidal con alpha adaptativo
        source_nonlinear = np.tanh(alpha * (2 * source_normalized - 1))
        source_nonlinear = (source_nonlinear + 1) / 2
        source_nonlinear = source_nonlinear * 255.0
        
        # Calcular estadísticas
        source_mean = np.mean(source_nonlinear)
        source_std = np.std(source_nonlinear)
        
        # Estadísticas de referencia
        channel_name = ['Red', 'Green', 'Blue'][channel]
        ref_mean = reference_stats[channel_name]['mean']
        ref_std = np.sqrt(reference_stats[channel_name]['variance'])
        
        # Evitar división por cero
        if source_std < 0.1:
            source_std = 0.1
        
        # Transformación lineal
        matched_data = (source_nonlinear - source_mean) * (ref_std / source_std) + ref_mean
        matched_data = np.clip(matched_data, 0, 255)
        matched_img[:, :, channel] = matched_data.astype(np.uint8)
    
    return matched_img

def match_histograms_adaptive(source_img, reference_stats, mask=None, adaptive_strength=1, preserve_ratio=0):
    """
    Adaptación de histograma con preservación de características y normalización adaptativa.
    
    Parámetros:
        source_img: Imagen fuente (en formato uint8, H x W x 3).
        reference_stats: Diccionario con medias y varianzas de referencia.
        mask: Máscara que indica píxeles a considerar (1) o ignorar (0).
        adaptive_strength: Control sobre qué tan fuerte es la adaptación (0-1).
        preserve_ratio: Ratio para preservar características originales (0-1).
    """
    matched_img = np.copy(source_img).astype(np.float32)
    
    # Si no hay máscara, crear una de todos unos
    if mask is None:
        mask = np.ones(source_img.shape[:2], dtype=bool)
    
    for channel in range(3):
        channel_name = ['Red', 'Green', 'Blue'][channel]
        ref_mean = reference_stats[channel_name]['mean']
        ref_std = np.sqrt(reference_stats[channel_name]['variance'])
        
        # Extraer datos del canal solo donde hay tejido (máscara=1)
        source_data = source_img[:, :, channel].astype(np.float32)
        tissue_pixels = source_data[mask == 1]
        
        # Si no hay suficientes píxeles de tejido, continuar
        if len(tissue_pixels) < 100:
            continue
        
        # Estadísticas del canal en el tejido
        source_mean = np.mean(tissue_pixels)
        source_std = np.std(tissue_pixels)
        if source_std < 0.1:
            source_std = 0.1
        
        # Crear LUTs para mapear valores con preservación
        lut = np.arange(256, dtype=np.float32)
        
        # Mapeo adaptativo que preserva valores extremos
        for i in range(256):
            # Valor normalizado (0-1)
            normalized_val = i / 255.0
            
            # Peso adaptativo según qué tan extremo es el valor
            extremity = 2 * abs(normalized_val - 0.5)  # 0 en el centro, 1 en los extremos
            adaptive_weight = adaptive_strength * (1 - extremity**2)
            
            # Valor preservado del original
            preserve_weight = preserve_ratio + (1 - preserve_ratio) * extremity
            
            # Mapeo lineal estándar
            standard_mapping = ((i - source_mean) * (ref_std / source_std) + ref_mean)
            
            # Combinar mapeo estándar y preservación
            final_mapping = (1 - preserve_weight) * standard_mapping + preserve_weight * i
            
            # Aplicar peso adaptativo
            lut[i] = (1 - adaptive_weight) * i + adaptive_weight * final_mapping
        
        # Asegurar valores en rango
        lut = np.clip(lut, 0, 255)
        
        # Aplicar LUT
        matched_img[:, :, channel] = cv2.LUT(source_img[:, :, channel], lut.astype(np.uint8))
    
    return matched_img.astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    para mejorar el contraste local.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def reinhard_normalization(source_img, reference_stats):
    """
    Implementa la normalización de color de Reinhard.
    Referencia: E. Reinhard, et al. "Color Transfer between Images"
    """
    # Convertir a espacio LAB
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Calcular estadísticas de la imagen fuente
    source_stats = {
        'L': {'mean': np.mean(source_lab[..., 0]), 'std': np.std(source_lab[..., 0])},
        'a': {'mean': np.mean(source_lab[..., 1]), 'std': np.std(source_lab[..., 1])},
        'b': {'mean': np.mean(source_lab[..., 2]), 'std': np.std(source_lab[..., 2])}
    }
    
    # Mapear estadísticas de RGB a LAB para referencia
    ref_lab_stats = {
        'L': {'mean': reference_stats['Red']['mean'] * 0.299 + 
                       reference_stats['Green']['mean'] * 0.587 + 
                       reference_stats['Blue']['mean'] * 0.114,
               'std': np.sqrt(reference_stats['Red']['variance'] * 0.299 + 
                             reference_stats['Green']['variance'] * 0.587 + 
                             reference_stats['Blue']['variance'] * 0.114)},
        'a': {'mean': 128, 'std': 10},  # Valores predeterminados aproximados
        'b': {'mean': 128, 'std': 10}   # Valores predeterminados aproximados
    }
    
    # Normalizar cada canal LAB
    for i, channel in enumerate(['L', 'a', 'b']):
        # Evitar divisiones por cero
        if source_stats[channel]['std'] < 0.1:
            source_stats[channel]['std'] = 0.1
            
        # Aplicar la normalización
        source_lab[..., i] = ((source_lab[..., i] - source_stats[channel]['mean']) * 
                              (ref_lab_stats[channel]['std'] / source_stats[channel]['std']) + 
                              ref_lab_stats[channel]['mean'])
    
    # Convertir de nuevo a RGB
    source_lab = np.clip(source_lab, 0, 255)
    return cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

def hybrid_normalization(source_img, reference_stats, mask=None, adaptive_strength=0.7, preserve_ratio=0.2):
    """
    Normalización híbrida que combina métodos adaptativos con preservación de características.
    """
    # Paso 1: Aplicar CLAHE para mejorar el contraste local
    clahe_result = apply_clahe(source_img)
    
    # Paso 2: Aplicar normalización adaptativa al resultado de CLAHE
    adaptive_result = match_histograms_adaptive(clahe_result, reference_stats, mask, 
                                               adaptive_strength, preserve_ratio)
    
    # Paso 3: Realizar un ajuste fino mediante Reinhard en áreas específicas
    if mask is not None and np.sum(mask) > 100:
        # Aplicar Reinhard solo en áreas de interés
        reinhard_result = reinhard_normalization(adaptive_result, reference_stats)
        
        # Combinar resultados basados en la máscara
        final_result = np.copy(adaptive_result)
        final_result[mask == 1] = (0.7 * adaptive_result[mask == 1] + 
                                  0.3 * reinhard_result[mask == 1])
        return final_result
    else:
        return adaptive_result

def normalize_image(source_img, reference_stats, method='adaptive', mask=None, 
                   adaptive_strength=0.7, preserve_ratio=0.2):
    """
    Función principal de normalización que selecciona el método apropiado.
    """
    if method == 'basic':
        return match_histograms_basic(source_img, reference_stats)
    elif method == 'adaptive':
        return match_histograms_adaptive(source_img, reference_stats, mask, 
                                       adaptive_strength, preserve_ratio)
    elif method == 'clahe':
        return apply_clahe(source_img)
    elif method == 'reinhard':
        return reinhard_normalization(source_img, reference_stats)
    elif method == 'hybrid':
        return hybrid_normalization(source_img, reference_stats, mask, 
                                  adaptive_strength, preserve_ratio)
    else:
        # Por defecto, usar el método adaptativo
        return match_histograms_adaptive(source_img, reference_stats, mask, 
                                       adaptive_strength, preserve_ratio)

def is_extreme_image(patch_array, extreme_threshold=0.95, min_variance=10):
    """
    Determina si una imagen tiene demasiados píxeles en los extremos (0 o 255)
    o muy poca varianza
    """
    # Calcular varianza global
    if patch_array.std() < min_variance:
        return True
    
    # Analizar histograma para cada canal
    for channel in range(3):
        hist = np.histogram(patch_array[:,:,channel].ravel(), bins=256, range=(0,256))[0]
        total_pixels = hist.sum()
        extreme_pixels = hist[0] + hist[-1]  # Píxeles en 0 y 255
        
        if extreme_pixels / total_pixels > extreme_threshold:
            return True
    
    return False

def extract_patches_from_slide(slide_path, patch_size=2048, level=0, reference_stats=None, 
                             extreme_threshold=0.95, min_variance=10, norm_method='adaptive',
                             denoise=False, visualize=False, tissue_threshold=0.8,
                             skip_background=False, adaptive_strength=0.7, preserve_ratio=0.2):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem

        # Directorios de salida - MANTENIENDO LA ESTRUCTURA ORIGINAL
        slide_output_dir = Path.cwd()/"result"/slide_name/"paches"
        os.makedirs(f"{slide_output_dir}/original", exist_ok=True)
        os.makedirs(f"{slide_output_dir}/matched", exist_ok=True)
        
        if visualize:
            viz_dir = f"{slide_output_dir}/visualizations"
            os.makedirs(viz_dir, exist_ok=True)

        # Obtener dimensiones
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level] if level > 0 else 1

        print(f"\nProcesando {slide_name} ({width}x{height} en nivel {level})")
        print(f"Método de normalización: {norm_method}")
        print(f"Filtros: extremos > {extreme_threshold*100}% | varianza < {min_variance}")

        patch_id = 0
        valid_patches = 0
        rejected_patches = 0
        
        for y in tqdm(range(0, height, patch_size), desc="Procesando filas"):
            for x in range(0, width, patch_size):
                actual_width = min(patch_size, width - x)
                actual_height = min(patch_size, height - y)

                # Solo recortes completos
                if actual_width == patch_size and actual_height == patch_size:
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                    patch = patch.convert("RGB")
                    patch_array = np.array(patch)
                    
                    # Verificar si el parche es válido
                    if is_extreme_image(patch_array, extreme_threshold, min_variance):
                        rejected_patches += 1
                        continue
                    
                    # Preprocesamiento: reducción de ruido
                    if denoise:
                        patch_array = denoise_image(patch_array)
                    
                    # Generar máscara de tejido si se requiere
                    mask = None
                    if skip_background:
                        mask = detect_tissue_mask(patch_array, tissue_threshold)
                        # Si no hay suficiente tejido, rechazar
                        if np.sum(mask) < 0.05 * mask.size:
                            rejected_patches += 1
                            continue
                    
                    # Guardar original - MANTENIENDO EL NOMBRE ORIGINAL
                    patch_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                    Image.fromarray(patch_array).save(slide_output_dir/"original"/patch_filename)
                    
                    # Aplicar normalización con el método seleccionado
                    matched_array = normalize_image(
                        patch_array, 
                        reference_stats, 
                        method=norm_method,
                        mask=mask,
                        adaptive_strength=adaptive_strength,
                        preserve_ratio=preserve_ratio
                    )
                    
                    # Guardar normalizado - MANTENIENDO EL NOMBRE ORIGINAL
                    # Nota: Se mantiene exactamente el formato de nombre del código original
                    matched_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}_matched.png"
                    Image.fromarray(matched_array).save(slide_output_dir/"matched"/matched_filename)
                    
                    # Generar visualización si se solicita
                    if visualize:
                        viz_path = f"{viz_dir}/{slide_name}_patch{patch_id:04d}_viz.png"
                        visualize_histograms(patch_array, matched_array, viz_path)
                    
                    patch_id += 1
                    valid_patches += 1

        print(f"Procesamiento completo. Parches válidos: {valid_patches} | Rechazados: {rejected_patches}")

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
    finally:
        if 'slide' in locals():
            slide.close()

def main():
    args = parse_args()
    input_path = Path(args.input_dir)

    # Cargar estadísticas de referencia
    try:
        with open(args.stats_json, 'r') as f:
            reference_stats = json.load(f)
        print(f"✅ Estadísticas cargadas de {args.stats_json}")
    except Exception as e:
        print(f"❌ Error al cargar el JSON: {e}")
        return

    # Buscar archivos MRXS
    if input_path.is_file() and input_path.suffix.lower() == ".mrxs":
        mrxs_files = [str(input_path)]
    elif input_path.is_dir():
        mrxs_files = glob.glob(os.path.join(args.input_dir, "*.mrxs"))
    else:
        print(f"❌ Ruta inválida: {args.input_dir}")
        return

    if not mrxs_files:
        print(f"❌ No se encontraron archivos MRXS en {args.input_dir}")
        return

    print(f"🔍 Encontrados {len(mrxs_files)} archivos MRXS")
    print(f"🧰 Método de normalización: {args.norm_method}")

    # Procesar cada archivo
    for slide_path in mrxs_files:
        extract_patches_from_slide(
            slide_path,
            patch_size=args.patch_size,
            level=args.level,
            reference_stats=reference_stats,
            extreme_threshold=args.extreme_threshold,
            min_variance=args.min_variance,
            norm_method=args.norm_method,
            denoise=args.denoise,
            visualize=args.visualize,
            tissue_threshold=args.tissue_threshold,
            skip_background=args.skip_background,
            adaptive_strength=args.adaptive_strength,
            preserve_ratio=args.preserve_ratio
        )

    print("\n✅ Procesamiento completado")

if __name__ == "__main__":
    main()