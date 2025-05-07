import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image
import cv2
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
import base64
import zlib
import uuid
import skimage.measure

# Import from mmseg and mmengine instead of mmcv
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope

def parse_args():
    parser = argparse.ArgumentParser(
        description='Recortar imágenes médicas MRXS y segmentar glomérulos'
    )
    parser.add_argument('--input_dir', type=str, required=True, help='Archivo MRXS o directorio con imágenes MRXS')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024, help='Stride (paso) para el recorte (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar (default: 0)')
    parser.add_argument('--extreme_threshold', type=float, default=0.95, help='Umbral para filtrar imágenes con píxeles extremos (0-1)')
    parser.add_argument('--min_variance', type=float, default=10.0, help='Varianza mínima para considerar la imagen válida')
    parser.add_argument('--denoise', action='store_true', help='Aplicar reducción de ruido')
    parser.add_argument('--tissue_threshold', type=float, default=0.8, help='Umbral para detección de tejido (0-1)')
    parser.add_argument('--skip_background', action='store_true', help='Ignorar parches con predominio de fondo')
    parser.add_argument('--save_roi', action='store_true', help='Guardar la ROI como una imagen de 1024x1024')
    parser.add_argument('--config_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mask2Former/mask2former_swin-b_kpis_isbi_768.py', 
                        help='Ruta del archivo de configuración del modelo')
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mmsegmentation/mask2former_swin-b_kpis_768/best_mDice_iter_6000.pth', 
                        help='Ruta del archivo de checkpoint del modelo')
    parser.add_argument('--save_original', action='store_true', help='Guardar parches originales además de las máscaras')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Transparencia de la máscara sobre la imagen original (0-1)')
    parser.add_argument('--mask_color', type=str, default='blue', help='Color para la máscara (red, green, blue, yellow, magenta, cyan)')
    parser.add_argument('--export_qupath', action='store_true', help='Exportar las máscaras a formato QuPath (.qpdata)')
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

def detect_roi(slide, tissue_threshold=0.5, preview_level=2):
    """
    Detecta la región de interés (ROI) donde se encuentran los riñones usando
    una versión de menor resolución de la imagen.
    
    Args:
        slide: Objeto OpenSlide
        tissue_threshold: Umbral para detección de tejido
        preview_level: Nivel de resolución para la detección (mayor número = menor resolución)
    
    Returns:
        tuple: (x_min, y_min, width, height) de la región de interés en coordenadas de nivel 0
    """
    # Asegurarnos de que el nivel de vista previa existe
    max_level = min(preview_level, len(slide.level_dimensions) - 1)
    
    # Obtener una versión de baja resolución de la imagen completa
    preview_width, preview_height = slide.level_dimensions[max_level]
    preview_img = slide.get_thumbnail((preview_width, preview_height))
    preview_array = np.array(preview_img.convert("RGB"))
    
    # Detectar máscara de tejido en la imagen de baja resolución
    tissue_mask = detect_tissue_mask(preview_array, threshold=tissue_threshold)
    
    # Encontrar los componentes conectados (las regiones de tejido)
    labeled, num_components = ndimage.label(tissue_mask)
    component_sizes = np.bincount(labeled.ravel())[1:] if num_components > 0 else []
    
    # Si no se detectan componentes, usar toda la imagen
    if len(component_sizes) == 0:
        print("No se detectaron componentes de tejido, usando toda la imagen")
        return 0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1]
    
    # Encontrar los índices de los componentes más grandes (potencialmente los riñones)
    # Ordenamos por tamaño y tomamos los 2 más grandes
    largest_components = np.argsort(component_sizes)[-2:] + 1  # +1 porque el fondo es 0
    
    # Crear una máscara solo con los componentes más grandes
    kidney_mask = np.isin(labeled, largest_components)
    
    # Encontrar el rectángulo delimitador (bounding box) de los componentes
    y_indices, x_indices = np.where(kidney_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        print("No se detectaron regiones de tejido válidas, usando toda la imagen")
        return 0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1]
    
    # Calcular coordenadas del rectángulo delimitador
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Añadir un margen del 10% alrededor del rectángulo
    margin_x = int((x_max - x_min) * 0.1)
    margin_y = int((y_max - y_min) * 0.1)
    
    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(preview_width, x_max + margin_x)
    y_max = min(preview_height, y_max + margin_y)
    
    # Convertir coordenadas al nivel 0 (resolución completa)
    downsample_factor = slide.level_downsamples[max_level]
    x_min_level0 = int(x_min * downsample_factor)
    y_min_level0 = int(y_min * downsample_factor)
    width_level0 = int((x_max - x_min) * downsample_factor)
    height_level0 = int((y_max - y_min) * downsample_factor)
    
    print(f"ROI detectado: x={x_min_level0}, y={y_min_level0}, ancho={width_level0}, alto={height_level0}")
    
    return x_min_level0, y_min_level0, width_level0, height_level0

def save_roi_image(slide, x_roi, y_roi, width_roi, height_roi, output_path, target_size=(1024, 1024)):
    """
    Guarda la región de interés (ROI) como una imagen de 1024x1024.
    
    Args:
        slide: Objeto OpenSlide
        x_roi, y_roi, width_roi, height_roi: Coordenadas del ROI en nivel 0
        output_path: Ruta donde guardar la imagen
        target_size: Tamaño deseado de la imagen de salida (ancho, alto)
    """
    # Determinar el nivel óptimo para leer la ROI
    # Queremos un nivel que nos dé una imagen similar al tamaño objetivo
    optimal_level = 0
    min_difference = float('inf')
    
    for level in range(len(slide.level_dimensions)):
        downsample = slide.level_downsamples[level]
        width_at_level = width_roi / downsample
        height_at_level = height_roi / downsample
        
        # Calcular qué tan cerca estamos del tamaño objetivo
        size_difference = abs(width_at_level - target_size[0]) + abs(height_at_level - target_size[1])
        
        if size_difference < min_difference:
            min_difference = size_difference
            optimal_level = level
    
    # Leer la ROI en el nivel óptimo
    downsample = slide.level_downsamples[optimal_level]
    x_level = int(x_roi / downsample)
    y_level = int(y_roi / downsample)
    width_level = int(width_roi / downsample)
    height_level = int(height_roi / downsample)
    
    # Leer la región
    roi_image = slide.read_region((x_roi, y_roi), optimal_level, (width_level, height_level))
    roi_image = roi_image.convert("RGB")
    
    # Redimensionar a 1024x1024 manteniendo la relación de aspecto
    # Primero calculamos el tamaño que respeta la relación de aspecto
    aspect_ratio = width_level / height_level
    
    if aspect_ratio > 1:  # Más ancho que alto
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Más alto que ancho o cuadrado
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    
    # Redimensionar la imagen manteniendo la relación de aspecto
    roi_image_resized = roi_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Crear un lienzo de 1024x1024 con fondo negro
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    
    # Calcular la posición para centrar la imagen
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    
    # Pegar la imagen redimensionada en el lienzo
    canvas.paste(roi_image_resized, (paste_x, paste_y))
    
    # Guardar la imagen
    canvas.save(output_path)
    print(f"ROI guardado como imagen {target_size[0]}x{target_size[1]} en {output_path}")

def initialize_model(config_path, ckpt_path, device='cuda:0'):
    """
    Inicializa el modelo Mask2Former para segmentación de glomérulos
    usando mmseg y mmengine
    """
    # Inicializar el ámbito predeterminado para mmseg
    init_default_scope('mmseg')
    
    # Cargar el modelo
    model = init_model(config_path, ckpt_path, device=device)
    
    # Definir el pipeline de prueba
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]
    
    # Asignar el pipeline al modelo
    model.cfg.test_pipeline = test_pipeline
    
    return model

def get_glomeruli_mask(result, target_height, target_width):
    """
    Convierte el resultado de la inferencia en una máscara binaria de glomérulos
    """
    # Obtener los logits de segmentación
    raw_logits = result.seg_logits.data
    
    # Obtener la clase con mayor probabilidad
    _, pred_mask = raw_logits.max(axis=0, keepdims=True)
    pred_mask = pred_mask.cpu().numpy()[0]
    
    # Clase de glomérulos (asumiendo que es la clase 1, ajusta según tu modelo)
    glomeruli_class = 1
    
    # Convertir a imagen binaria
    binary_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    binary_mask[pred_mask == glomeruli_class] = 255
    
    # Redimensionar al tamaño objetivo si es necesario
    if binary_mask.shape[0] != target_height or binary_mask.shape[1] != target_width:
        binary_mask = cv2.resize(binary_mask, (target_width, target_height), 
                                interpolation=cv2.INTER_NEAREST)
    
    return binary_mask

def get_mask_color(color_name):
    """
    Convierte un nombre de color a valores BGR para OpenCV
    """
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (0, 255, 255),
        'magenta': (255, 0, 255),
        'cyan': (255, 255, 0),
    }
    return colors.get(color_name.lower(), (255, 0, 0))  # Por defecto rojo

def create_overlay_image(original_image, mask, alpha=0.5, color_name='blue'):
    """
    Crea una imagen con la máscara superpuesta semitransparente en la imagen original.
    Asume que `original_image` ya está en formato BGR (sin conversión).
    
    Args:
        original_image: Imagen original en formato numpy array (BGR)
        mask: Máscara binaria (0-255)
        alpha: Nivel de transparencia de la máscara (0-1)
        color_name: Nombre del color de la máscara
    
    Returns:
        Imagen compuesta con la máscara superpuesta a la imagen original (RGB)
    """
    # Log: Verificar tipo y formato de entrada
    
    if isinstance(original_image, Image.Image):
        original_bgr = np.array(original_image)[:, :, ::-1]  # Convertir PIL RGB → BGR
    else:
        original_bgr = original_image.copy()
    
    
    # Crear una máscara de color
    mask_bgr = np.zeros_like(original_bgr)
    color_bgr = get_mask_color(color_name)
    
    # Aplicar el color solo donde la máscara es > 0
    mask_bool = mask > 0
    mask_bgr[mask_bool] = color_bgr
    
    # Log: Verificar un pixel sin máscara (no debería cambiar)
    sample_x, sample_y = 10, 10  # Posición de prueba (ajustar si la máscara lo cubre)
    original_pixel = original_bgr[sample_y, sample_x]
    
    # Crear imagen superpuesta (sin alterar zonas sin máscara)
    overlay = cv2.addWeighted(original_bgr, 1.0, mask_bgr, alpha, 0)
    
    # Log: Comprobar si el pixel de prueba cambió (no debería si no hay máscara)
    overlay_pixel = overlay[sample_y, sample_x]
    if not mask_bool[sample_y, sample_x]:
        assert np.allclose(original_pixel, overlay_pixel), "¡Error: El pixel sin máscara fue alterado!"
    
    # Convertir a RGB para visualización (opcional, si necesitas RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay_rgb

def save_composite_mask(masks_dict, original_dict, output_path, roi_dimensions, 
                       alpha=0.5, color_name='blue'):
    """
    Guarda una imagen compuesta de todas las máscaras de glomérulos superpuestas a
    las imágenes originales, con el tamaño exacto de la ROI y sin márgenes entre parches.
    
    Args:
        masks_dict: Diccionario con las coordenadas (x, y) como claves y las máscaras como valores
        original_dict: Diccionario con las coordenadas (x, y) como claves y las imágenes originales como valores
        output_path: Ruta donde guardar la imagen compuesta
        roi_dimensions: Tupla (width, height) con las dimensiones de la ROI original
        alpha: Nivel de transparencia de la máscara
        color_name: Color de la máscara superpuesta
    """
    # Extraer las dimensiones de la ROI
    roi_width, roi_height = roi_dimensions
    
    # Crear una imagen del tamaño exacto de la ROI para la composición
    composite = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
    
    if masks_dict:
        # Obtener el tamaño de los parches
        first_mask = next(iter(masks_dict.values()))
        patch_height, patch_width = first_mask.shape[:2]
        
        # Encontrar las coordenadas mínimas para usar como origen (0,0) en la imagen compuesta
        min_x = min([x for x, y in masks_dict.keys()])
        min_y = min([y for x, y in masks_dict.keys()])
        
        # Colocar cada parche con su máscara superpuesta en la posición exacta sin escalar
        for (x, y), mask in masks_dict.items():
            if (x, y) in original_dict:
                # Calcular posición relativa al origen de la ROI
                rel_x = x - min_x
                rel_y = y - min_y
                
                # Obtener la imagen original correspondiente
                original = original_dict[(x, y)]
                
                # Crear la imagen con la máscara superpuesta
                overlay = create_overlay_image(original, mask, alpha, color_name)
                
                # Asegurarse de que no excede los límites de la imagen compuesta
                y_end = min(rel_y + patch_height, roi_height)
                x_end = min(rel_x + patch_width, roi_width)
                
                # Calcular cuánto del parche se va a copiar
                patch_y_end = y_end - rel_y
                patch_x_end = x_end - rel_x
                
                # Copiar la porción de la imagen con máscara a la composición
                try:
                    if rel_x < roi_width and rel_y < roi_height:
                        composite[rel_y:y_end, rel_x:x_end] = overlay[:patch_y_end, :patch_x_end]
                except Exception as e:
                    print(f"Error al componer imagen en posición ({rel_x},{rel_y}): {e}")
    
    # Guardar la imagen compuesta
    cv2.imwrite(str(output_path), composite)
    print(f"Imagen compuesta guardada en {output_path} con dimensiones {roi_width}x{roi_height}")

def extract_patches_and_predict(slide_path, model, patch_size=2048, level=0, 
                              extreme_threshold=0.95, min_variance=10,
                              denoise=False, tissue_threshold=0.8,
                              skip_background=False, save_roi=False,
                              save_original=False, overlay_alpha=0.5,
                              mask_color='blue', export_qupath=True):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem

        # Directorios de salida en la ubicación específica
        # Extraer los valores de R y S del nombre del slide (ejemplo: slide-2023-02-18T08-07-19-R3-S11)
        if "R" in slide_name and "S" in slide_name:
            # Buscar el patrón R seguido de números
            r_match = slide_name.split('-R')[-1].split('-')[0]
            # Buscar el patrón S seguido de números
            s_match = slide_name.split('-S')[-1].split('-')[0]
            
            # Construir la ruta con los valores específicos de R y S
            slide_output_dir = Path(f"/mnt/work/users/bernat.olle/Results/R{r_match}/S{s_match}")
        else:
            # Si no encuentra el patrón, usar una ruta por defecto
            slide_output_dir = Path(f"/mnt/work/users/bernat.olle/Results/{slide_name}")
        
        # Crear directorios para máscaras, originales y superposiciones
        masks_output_dir = slide_output_dir / "masks"
        os.makedirs(masks_output_dir, exist_ok=True)
        
        original_output_dir = slide_output_dir / "original"
        os.makedirs(original_output_dir, exist_ok=True)
        
        overlay_output_dir = slide_output_dir / "overlay"
        os.makedirs(overlay_output_dir, exist_ok=True)
        
        # Crear directorio para archivos QuPath si es necesario
        qupath_output_dir = slide_output_dir / "qupath"
        if export_qupath:
            os.makedirs(qupath_output_dir, exist_ok=True)

        # Detectar región de interés (ROI) donde están los riñones
        print("Detectando región de interés (ROI)...")
        x_roi, y_roi, width_roi, height_roi = detect_roi(slide, tissue_threshold=0.5, preview_level=2)
        
        # Guardar la ROI como una imagen de 1024x1024 si se solicita
        if save_roi:
            roi_output_path = slide_output_dir / f"{slide_name}_ROI_1024x1024.png"
            save_roi_image(slide, x_roi, y_roi, width_roi, height_roi, roi_output_path)

        # Obtener dimensiones
        downsample = slide.level_downsamples[level] if level > 0 else 1

        print(f"\nProcesando {slide_name} (ROI: {width_roi}x{height_roi} en nivel {level})")
        print(f"Filtros: extremos > {extreme_threshold*100}% | varianza < {min_variance}")

        patch_id = 0
        valid_patches = 0
        rejected_patches = 0
        
        # Diccionarios para almacenar las máscaras y sus posiciones
        masks_dict = {}
        original_dict = {}
        
        # Calcular coordenadas de ROI ajustadas al nivel deseado
        x_roi_level = int(x_roi / downsample) if level > 0 else x_roi
        y_roi_level = int(y_roi / downsample) if level > 0 else y_roi
        width_roi_level = int(width_roi / downsample) if level > 0 else width_roi
        height_roi_level = int(height_roi / downsample) if level > 0 else height_roi
        
        # Iterar solo sobre la región de interés
        for y in tqdm(range(y_roi_level, y_roi_level + height_roi_level, patch_size), desc="Procesando filas"):
            for x in range(x_roi_level, x_roi_level + width_roi_level, patch_size):
                actual_width = min(patch_size, x_roi_level + width_roi_level - x)
                actual_height = min(patch_size, y_roi_level + height_roi_level - y)

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
                    
                    # Verificar si hay suficiente tejido si se requiere
                    if skip_background:
                        mask = detect_tissue_mask(patch_array, tissue_threshold)
                        # Si no hay suficiente tejido, rechazar
                        if np.sum(mask) < 0.05 * mask.size:
                            rejected_patches += 1
                            continue
                    
                    # Preprocesamiento: reducción de ruido si se solicita
                    if denoise:
                        patch_array = denoise_image(patch_array)
                    
                    # Guardar parche original si se solicita
                    if save_original:
                        orig_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                        Image.fromarray(patch_array).save(original_output_dir / orig_filename)
                    
                    # Almacenar imagen original para composición final
                    original_dict[(x, y)] = patch_array
                    
                    # Convertir de RGB a BGR para OpenCV
                    patch_bgr = cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR)
                    
                    # Inferencia del modelo para segmentar glomérulos
                    try:
                        # Usando mmengine y mmseg en lugar de mmcv
                        result = inference_model(model, patch_bgr)
                        
                        # Obtener la máscara de glomérulos
                        glomeruli_mask = get_glomeruli_mask(result, patch_size, patch_size)
                        
                        # Guardar la máscara binaria
                        mask_filename = f"{slide_name}_mask{patch_id:04d}_x{x}_y{y}.png"
                        cv2.imwrite(str(masks_output_dir / mask_filename), glomeruli_mask)
                        
                        # Crear y guardar la imagen con la máscara superpuesta
                        overlay_image = create_overlay_image(patch_array, glomeruli_mask, overlay_alpha, mask_color)
                        overlay_filename = f"{slide_name}_overlay{patch_id:04d}_x{x}_y{y}.png"
                        cv2.imwrite(str(overlay_output_dir / overlay_filename), overlay_image)
                        
                        # Almacenar la máscara para la imagen compuesta
                        masks_dict[(x, y)] = glomeruli_mask
                        
                        patch_id += 1
                        valid_patches += 1
                    except Exception as e:
                        print(f"Error en la predicción del parche {patch_id}: {e}")
                        rejected_patches += 1

        # Generar y guardar la imagen compuesta del tamaño exacto de la ROI
        composite_path = slide_output_dir / f"{slide_name}_glomeruli_composite.png"
        save_composite_mask(masks_dict, original_dict, composite_path, 
                           roi_dimensions=(width_roi_level, height_roi_level),
                           alpha=overlay_alpha, color_name=mask_color)
        
        # Exportar a formato QuPath si se solicita
        if export_qupath and masks_dict:
            qpdata_path = qupath_output_dir / f"{slide_name}_glomeruli.qpdata"
            generate_qpdata(masks_dict, original_dict, slide_path, qpdata_path, 
                           downsample, level, patch_size)

        print(f"Procesamiento completo. Parches válidos: {valid_patches} | Rechazados: {rejected_patches}")

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
    finally:
        if 'slide' in locals():
            slide.close()

def generate_qpdata(masks_dict, original_dict, slide_path, output_path, downsample, level, patch_size):
    """
    Genera un archivo .qpdata para QuPath con todas las máscaras de glomérulos detectadas.
    
    Args:
        masks_dict: Diccionario con las coordenadas (x, y) como claves y las máscaras como valores
        original_dict: Diccionario con las coordenadas (x, y) como claves y las imágenes originales
        slide_path: Ruta al archivo MRXS original
        output_path: Ruta donde guardar el archivo .qpdata
        downsample: Factor de escala del nivel procesado
        level: Nivel de la imagen procesada
        patch_size: Tamaño de los parches procesados
    """
    if not masks_dict:
        print("No hay máscaras para exportar a QuPath")
        return
    
    # Nombre del archivo slide
    slide_name = Path(slide_path).stem
    slide_path_abs = str(Path(slide_path).resolve())
    
    # Crear la estructura de datos para QuPath con referencia a la imagen
    qupath_data = {
        "type": "io.github.qupath.core.objects.PathAnnotationObject",
        "id": str(uuid.uuid4()),
        "name": "Glomeruli segmentation",
        "color": -16776961,  # Color azul en RGB int
        "colorRGB": -16776961,
        "classProbability": 1.0,
        "locked": False,
        "readme": f"Glomeruli segmentation generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "annotations": [],
        "imageData": {
            "serverBuilder": {
                "builderType": "openslide",
                "uri": slide_path_abs
            },
            "entryID": 1,
            "imageName": slide_name
        }
    }
    
    # Contar cuántas máscaras tenemos para procesar
    total_masks = len(masks_dict)
    print(f"Procesando {total_masks} máscaras para exportar a QuPath...")
    
    # Contador para ROIs encontrados
    total_rois = 0
    
    # Para cada máscara, extraer los contornos y convertirlos a coordenadas de QuPath
    for (x, y), mask in masks_dict.items():
        # Convertir la máscara binaria a formato uint8 para OpenCV
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0).astype(np.uint8) * 255
        else:
            binary_mask = mask.copy()
        
        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcular las coordenadas absolutas en el nivel 0 (resolución completa)
        x_abs = int(x * downsample)
        y_abs = int(y * downsample)
        
        # Procesar cada contorno encontrado
        for contour in contours:
            # Filtrar contornos muy pequeños (posible ruido)
            if cv2.contourArea(contour) < 100:  # Ajustar este valor según sea necesario
                continue
            
            # Simplificar el contorno para reducir el número de puntos
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convertir contorno a formato QuPath (lista de vértices)
            vertices = []
            for point in approx:
                # Convertir coordenadas del parche a coordenadas absolutas del slide
                px, py = point[0]
                x_global = x_abs + px
                y_global = y_abs + py
                
                vertices.append({
                    "x": float(x_global),
                    "y": float(y_global)
                })
            
            # Si el contorno es válido, agregar a la lista de anotaciones
            if len(vertices) >= 3:  # Un polígono necesita al menos 3 vértices
                roi_data = {
                    "type": "io.github.qupath.core.objects.PathAnnotationObject",
                    "id": str(uuid.uuid4()),
                    "name": "Glomerulus",
                    "color": -16776961,  # Azul
                    "colorRGB": -16776961,
                    "classProbability": 1.0,
                    "locked": False,
                    "roi": {
                        "type": "io.github.qupath.core.objects.classes.ROI",
                        "name": "Polygon",
                        "vertices": vertices,
                        "closed": True
                    },
                    "measurements": []
                }
                
                qupath_data["annotations"].append(roi_data)
                total_rois += 1
    
    # Escribir el archivo .qpdata
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qupath_data, f, indent=2)
    
    print(f"Exportación a QuPath completada: {total_rois} ROIs de glomérulos guardados en {output_path}")

    # También crear un archivo de proyecto QuPath mínimo
    project_file = Path(output_path).parent / f"{slide_name}.qpproj"
    
    project_data = {
        "version": "0.3.2",
        "uri": str(project_file.resolve()),
        "name": f"Glomeruli Project - {slide_name}",
        "description": "Proyecto de segmentación automática de glomérulos",
        "creationTimestamp": datetime.now().isoformat(),
        "modificationTimestamp": datetime.now().isoformat(),
        "images": [{
            "serverBuilder": {
                "builderType": "openslide",
                "uri": slide_path_abs
            },
            "entryID": 1,
            "randomizedName": str(uuid.uuid4()),
            "name": slide_name,
            "metadata": {
                "openslide.vendor": "3DHISTECH",
                "openslide.level-count": "1"
            },
            "objectData": str(output_path.resolve())  # Referencia al archivo de anotaciones
        }]
    }
    
    with open(project_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=2)
    
    print(f"Archivo de proyecto QuPath generado en {project_file}")
    
    return output_path

def main():
    args = parse_args()
    input_path = Path(args.input_dir)

    # Inicializar el modelo
    print("Inicializando modelo Mask2Former...")
    model = initialize_model(args.config_path, args.ckpt_path)
    print("Modelo cargado correctamente")

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

    # Procesar cada archivo
    for slide_path in mrxs_files:
        extract_patches_and_predict(
            slide_path,
            model,
            patch_size=args.patch_size,
            level=args.level,
            extreme_threshold=args.extreme_threshold,
            min_variance=args.min_variance,
            denoise=args.denoise,
            tissue_threshold=args.tissue_threshold,
            skip_background=args.skip_background,
            save_roi=args.save_roi,
            save_original=args.save_original,
            overlay_alpha=args.overlay_alpha,
            mask_color=args.mask_color,
            export_qupath=args.export_qupath
        )

    print("\n✅ Procesamiento completado")

if __name__ == "__main__":
    main()