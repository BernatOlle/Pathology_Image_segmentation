import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image
import re
import cv2
from scipy import ndimage
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
import base64
import sys
import zlib
import uuid
import skimage.measure
import time  # Añadido para medición de tiempo
from datetime import timedelta, datetime  # Add this line
from logger import setup_logger, get_logger
from contar_glomerulos import GlomeruliParameterCalibrator
sys.path.append('..')
from bowman.bowman2 import GlomeruliWhiteAreaAnalyzer

# Import from mmseg and mmengine instead of mmcv
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope




def parse_args():
    parser = argparse.ArgumentParser(
        description='Recortar imágenes médicas MRXS y segmentar glomérulos'
    )
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Ruta al archivo MRXS (ej: /mnt/work/datasets/BKidney/CROC/slide-2023-02-18T08-00-55-R3-S6.mrxs)')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--overlap', type=int, default=1024, help='Stride (paso) para el recorte (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar (default: 0)')
    parser.add_argument('--extreme_threshold', type=float, default=0.95, help='Umbral para filtrar imágenes con píxeles extremos (0-1)')
    parser.add_argument('--min_variance', type=float, default=10.0, help='Varianza mínima para considerar la imagen válida')
    parser.add_argument('--denoise', action='store_true', help='Aplicar reducción de ruido')
    parser.add_argument('--tissue_threshold', type=float, default=0.8, help='Umbral para detección de tejido (0-1)')
    parser.add_argument('--skip_background', action='store_true', help='Ignorar parches con predominio de fondo')
    parser.add_argument('--background_threshold', type=float, default=0.85, 
                        help='Umbral para considerar un parche como fondo (0-1)')
    parser.add_argument('--extreme_pixel_ratio', type=float, default=0.9, 
                        help='Proporción de píxeles extremos para considerar un parche como fondo (0-1)')
    parser.add_argument('--save_roi', action='store_true', help='Guardar la ROI como una imagen de 1024x1024')
    parser.add_argument('--config_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mask2Former/mask2former_swin-b_kpis_isbi_768.py', 
                        help='Ruta del archivo de configuración del modelo')
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mmsegmentation/mask2former_swin-b_kpis_768/best_mDice_iter_21000.pth', 
                        help='Ruta del archivo de checkpoint del modelo')
    parser.add_argument('--save_original', action='store_true', help='Guardar parches originales')
    parser.add_argument('--save_mask', action='store_true', help='Guardar máscaras de segmentación')
    parser.add_argument('--save_composite', action='store_true', help='Guardar WSI completa de mascaras')
    parser.add_argument('--save_geojson', action='store_true', help='Guardar máscaras en formato GeoJSON para QuPath')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Transparencia de la máscara sobre la imagen original (0-1)')
    parser.add_argument('--mask_color', type=str, default='blue', help='Color para la máscara (red, green, blue, yellow, magenta, cyan)')
    return parser.parse_args()

def detect_tissue_mask(img, threshold=0.8):
    """
    Detecta una máscara de tejido en la imagen, eliminando el fondo.
    Devuelve una máscara binaria donde 1=tejido, 0=fondo.
    """
    # Convertir a escala de grises
    logger = get_logger()
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
    logger = get_logger()
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
    logger = get_logger()
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
        logger.info("No se detectaron componentes de tejido, usando toda la imagen")
        return 0, 0, slide.level_dimensions[0][0], slide.level_dimensions[0][1]
    
    # Encontrar los índices de los componentes más grandes (potencialmente los riñones)
    # Ordenamos por tamaño y tomamos los 2 más grandes
    largest_components = np.argsort(component_sizes)[-2:] + 1  # +1 porque el fondo es 0
    
    # Crear una máscara solo con los componentes más grandes
    kidney_mask = np.isin(labeled, largest_components)
    
    # Encontrar el rectángulo delimitador (bounding box) de los componentes
    y_indices, x_indices = np.where(kidney_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        logger.info("No se detectaron regiones de tejido válidas, usando toda la imagen")
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
    
    logger.info(f"ROI detectado: x={x_min_level0}, y={y_min_level0}, ancho={width_level0}, alto={height_level0}")
    
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
    logger = get_logger()
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
    logger.info(f"ROI guardado como imagen {target_size[0]}x{target_size[1]} en {output_path}")

def initialize_model(config_path, ckpt_path, device='cuda:0'):
    """
    Inicializa el modelo Mask2Former para segmentación de glomérulos
    usando mmseg y mmengine. Si el checkpoint especificado no existe,
    busca el archivo best_mDice_iter_xx.pth con el número más alto en la misma carpeta.
    """
    # Inicializar el ámbito predeterminado para mmseg
    logger = get_logger()
    init_default_scope('mmseg')
    
    # Verificar si el checkpoint existe
    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint {ckpt_path} no encontrado. Buscando alternativa...")
        
        # Obtener el directorio donde buscar
        ckpt_dir = os.path.dirname(ckpt_path)
        
        # Buscar todos los archivos que coincidan con el patrón best_mDice_iter_*.pth
        pattern = os.path.join(ckpt_dir, "best_mDice_iter_*.pth")
        matches = glob.glob(pattern)
        
        if matches:
            # Extraer el número de iteración de cada archivo y ordenar
            iter_files = []
            for match in matches:
                filename = os.path.basename(match)
                iter_match = re.search(r'best_mDice_iter_(\d+)\.pth', filename)
                if iter_match:
                    iter_num = int(iter_match.group(1))
                    iter_files.append((iter_num, match))
            
            # Ordenar por número de iteración (de mayor a menor)
            iter_files.sort(reverse=True)
            
            if iter_files:
                # Usar el archivo con el número de iteración más alto
                best_ckpt = iter_files[0][1]
                logger.info(f"Usando checkpoint alternativo: {best_ckpt}")
                ckpt_path = best_ckpt
            else:
                logger.error("No se encontraron checkpoints alternativos.")
        else:
            logger.error(f"No se encontraron archivos que coincidan con {pattern}")
    
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
    logger = get_logger()
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
                      alpha=0.5, color_name='blue', target_size=(1024, 1024), overlap=512, 
                      show_patch_borders=False, use_absolute_coords=True, downsample_factor=2):
    """
    Crea una máscara compuesta del área mínima que contiene todos los parches con glomérulos detectados.
    Calcula el bounding box correcto sin margen adicional.
    """
    import numpy as np
    import cv2
    
    if not masks_dict:
        return {
            'composite_mask': None,
            'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
            'downsample_info': {'factor': int(downsample_factor), 'downsampled_width': 0, 'downsampled_height': 0},
            'patch_info': []
        }
    
    try:
        # Obtener dimensiones de los parches
        first_mask = next(iter(masks_dict.values()))
        original_patch_height, original_patch_width = first_mask.shape[:2]
        
        # Calcular tamaños con downsampling
        patch_height = original_patch_height // downsample_factor
        patch_width = original_patch_width // downsample_factor
        downsampled_overlap = overlap // downsample_factor
        
        # Encontrar límites de los parches que contienen glomérulos
        patches_with_glomeruli = []
        
        for (x, y), mask in masks_dict.items():
            # Aplicar downsampling
            if downsample_factor > 1:
                h, w = mask.shape[:2]
                new_h = max(1, h // downsample_factor)
                new_w = max(1, w // downsample_factor)
                downsampled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                downsampled_mask = mask
            
            # Verificar si hay glomérulos en esta máscara
            if np.any(downsampled_mask > 0):
                # Coordenadas ajustadas al downsampling
                adj_x = x // downsample_factor
                adj_y = y // downsample_factor
                patches_with_glomeruli.append((adj_x, adj_y, downsampled_mask))
        
        # Verificar que se encontraron parches con glomérulos
        if not patches_with_glomeruli:
            return {
                'composite_mask': None,
                'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                'downsample_info': {'factor': int(downsample_factor), 'downsampled_width': 0, 'downsampled_height': 0},
                'patch_info': []
            }
        
        # Calcular bounding box que incluya todos los parches completos
        min_x = min(x for x, y, _ in patches_with_glomeruli)
        max_x = max(x for x, y, _ in patches_with_glomeruli)
        min_y = min(y for x, y, _ in patches_with_glomeruli)
        max_y = max(y for x, y, _ in patches_with_glomeruli)
        
        # El bounding box debe incluir el parche completo más extremo
        bbox_min_x = min_x
        bbox_min_y = min_y
        bbox_max_x = max_x + patch_width - 1
        bbox_max_y = max_y + patch_height - 1
        
        # Dimensiones finales
        bbox_width = bbox_max_x - bbox_min_x + 1
        bbox_height = bbox_max_y - bbox_min_y + 1
        
        # Crear imagen compuesta
        composite = np.zeros((bbox_height, bbox_width), dtype=np.uint8)
        patch_info = []
        
        # Llenar la imagen compuesta
        for (original_x, original_y), mask in masks_dict.items():
            # Aplicar downsampling
            if downsample_factor > 1:
                h, w = mask.shape[:2]
                new_h = max(1, h // downsample_factor)
                new_w = max(1, w // downsample_factor)
                downsampled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                downsampled_mask = mask
            
            # Verificar si hay glomérulos en esta máscara
            if not np.any(downsampled_mask > 0):
                continue
            
            # Coordenadas ajustadas
            x = original_x // downsample_factor
            y = original_y // downsample_factor
            
            # Posición relativa en el bounding box
            rel_x = x - bbox_min_x
            rel_y = y - bbox_min_y
            
            # Colocar máscara completa
            mask_h, mask_w = downsampled_mask.shape[:2]
            end_y = rel_y + mask_h
            end_x = rel_x + mask_w
            
            # Verificar que el parche cabe completamente (debería ser siempre true)
            if end_y <= bbox_height and end_x <= bbox_width and rel_x >= 0 and rel_y >= 0:
                # Crear máscara binaria
                binary_mask = (downsampled_mask > 0).astype(np.uint8) * 255
                
                # Aplicar máscara (usar OR para combinar)
                composite[rel_y:end_y, rel_x:end_x] = np.maximum(
                    composite[rel_y:end_y, rel_x:end_x], 
                    binary_mask
                )
                
                # Guardar info del parche
                patch_info.append({
                    'original_coords': (int(original_x), int(original_y)),
                    'downsampled_coords': (int(x), int(y)),
                    'relative_coords': (int(rel_x), int(rel_y)),
                    'glomeruli_pixels': int(np.sum(binary_mask > 0))
                })
        
        # Agregar bordes de parches si se solicita
        if show_patch_borders:
            final_image = np.zeros((bbox_height, bbox_width, 3), dtype=np.uint8)
            final_image[:, :, 0] = composite  # Máscaras en canal azul
            
            # Dibujar bordes de los parches
            for info in patch_info:
                rel_x, rel_y = info['relative_coords']
                # Dibujar rectángulo del parche completo
                cv2.rectangle(final_image, (rel_x, rel_y), 
                            (rel_x + patch_width - 1, rel_y + patch_height - 1), 
                            (0, 128, 0), 1)
        else:
            final_image = composite
        
        # Guardar imagen
        try:
            cv2.imwrite(str(output_path), final_image)
        except Exception as e:
            pass
        
        # Coordenadas en escala original para el resultado
        result = {
            'composite_mask': final_image,
            'bounding_box': {
                'x': bbox_min_x * downsample_factor,
                'y': bbox_min_y * downsample_factor,
                'width': bbox_width * downsample_factor,
                'height': bbox_height * downsample_factor
            },
            'downsample_info': {
                'factor': int(downsample_factor),
                'downsampled_width': bbox_width,
                'downsampled_height': bbox_height
            },
            'patch_info': patch_info
        }
        
        return result
        
    except Exception as e:
        return {
            'composite_mask': None,
            'bounding_box': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
            'downsample_info': {'factor': int(downsample_factor), 'downsampled_width': 0, 'downsampled_height': 0},
            'patch_info': []
        }
        
def is_background_patch(patch_array, background_threshold=0.85, extreme_pixel_ratio=0.9):
    """
    Determina si un parche es mayoritariamente fondo (blanco, negro o mezcla de ambos).
    
    Args:
        patch_array: Numpy array de la imagen en formato RGB
        background_threshold: Umbral para considerar un parche como fondo (0-1)
        extreme_pixel_ratio: Proporción de píxeles extremos para considerar como blanco/negro
    
    Returns:
        bool: True si el parche es mayoritariamente fondo, False en caso contrario
    """
    logger = get_logger()
    
    if patch_array is None or patch_array.size == 0:
        return True
    
    # Convertir a escala de grises para análisis
    if len(patch_array.shape) == 3:
        gray = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = patch_array.copy()
    
    # Calcular histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / gray.size
    
    # Calcular proporción de píxeles extremos (cercanos a negro o blanco)
    dark_pixels = sum(hist[:25])    # Píxeles en el rango 0-24
    light_pixels = sum(hist[230:])  # Píxeles en el rango 230-255
    extreme_pixels = dark_pixels + light_pixels
    
    # Comprobar si mayoría de píxeles son negros o blancos
    if extreme_pixels > extreme_pixel_ratio:
        return True
    
    # Analizar varianza y distribución de píxeles
    mean_val = gray.mean()
    std_val = gray.std()
    
    # Calcular porcentaje de píxeles dentro del rango cercano a la media
    # (esto identifica imágenes con poca variación tonal)
    lower_bound = max(0, mean_val - std_val)
    upper_bound = min(255, mean_val + std_val)
    
    # Crear máscara para píxeles en ese rango
    mask = (gray >= lower_bound) & (gray <= upper_bound)
    pixels_in_range = np.sum(mask) / gray.size
    
    # Si un alto porcentaje de píxeles están en un rango estrecho alrededor de la media,
    # probablemente sea un fondo uniforme
    if pixels_in_range > background_threshold:
        return True
    
    # Opcionalmente: detectar también patrones repetitivos típicos de fondo
    # (como cuadrículas o patrones de escáner)
    
    return False


def save_level_image(slide, output_dir, level=7, target_size=None, bounds_x=0, bounds_y=0, bounds_width=None, bounds_height=None,slide_name=None):
    logger = get_logger()
    level_downsample = slide.level_downsamples[level]
    region_width = int(bounds_width / level_downsample)
    region_height = int(bounds_height / level_downsample)
    region = slide.read_region(
        (bounds_x, bounds_y),  # Start at the bounds offset
        level,                  # Pyramid level
        (region_width, region_height)  # Size to read
    )
    
    # Convert to RGB
    region_rgb = region.convert('RGB')
    
    # Save the image
    output_path = os.path.join(output_dir, f"{slide_name}_level{level}.png")
    region_rgb.save(output_path)
    logger.info(f"\nSaved level {level} image to: {output_path}")
    
    # Save a thumbnail version if it's a large image
    if max(region_width, region_height) > 1024:
        thumb = region_rgb.copy()
        thumb.thumbnail((1024, 1024))
        thumb_path = os.path.join(output_dir, f"{slide_name}_level{level}_thumbnail.png")
        thumb.save(thumb_path)
        logger.info(f"Saved thumbnail to: {thumb_path}")

# Alternativa más simple si prefieres modificar directamente el array
def add_overlap_lines_to_array(image_array, overlap=32, line_color=(255, 0, 255), line_width=6):
    """
    Añade líneas de overlap directamente al array de la imagen
    """
    height, width = image_array.shape[:2]
    line_distance = overlap // 2
    
    # Crear una copia para no modificar el original
    result = image_array.copy()
    
    # Líneas horizontales
    for i in range(line_width):
        # Superior
        if line_distance + i < height:
            result[line_distance + i, :] = line_color
        # Inferior  
        if height - line_distance - i - 1 >= 0:
            result[height - line_distance - i - 1, :] = line_color
    
    # Líneas verticales
    for i in range(line_width):
        # Izquierda
        if line_distance + i < width:
            result[:, line_distance + i] = line_color
        # Derecha
        if width - line_distance - i - 1 >= 0:
            result[:, width - line_distance - i - 1] = line_color
    
    return result

def extract_patches_and_predict(slide_path, model, patch_size=2048, level=0, 
                              extreme_threshold=0.95, min_variance=10,
                              denoise=False, tissue_threshold=0.8,
                              skip_background=False, save_roi=False,
                              save_original=False, save_mask=False, 
                              save_composite=False, save_geojson=False,
                              overlay_alpha=0.5, mask_color='blue',
                              background_threshold=0.85, extreme_pixel_ratio=0.9, overlap=512):
    logger = get_logger()
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem
        

        # Get the full slide dimensions
        full_width = slide.dimensions[0]
        full_height = slide.dimensions[1]

        # Get the bounds/region properties (this is what QuPath is using)
        bounds_x = int(slide.properties.get('openslide.bounds-x', 0))
        bounds_y = int(slide.properties.get('openslide.bounds-y', 0))
        bounds_width = int(slide.properties.get('openslide.bounds-width', full_width))
        bounds_height = int(slide.properties.get('openslide.bounds-height', full_height))

        logger.info(f"Full slide dimensions: {full_width} x {full_height}")
        logger.info(f"Tissue bounds: {bounds_width} x {bounds_height} at position ({bounds_x}, {bounds_y})")

        # Directorios de salida en la ubicación específica
        if "R" in slide_name and "S" in slide_name:
            r_match = slide_name.split('-R')[-1].split('-')[0]
            s_match = slide_name.split('-S')[-1].split('-')[0]
            slide_output_dir = Path(f"/mnt/work/users/bernat.olle/Results/R{r_match}/S{s_match}")
        else:
            slide_output_dir = Path(f"/mnt/work/users/bernat.olle/Results/{slide_name}")
        logger.info(f"Directorio de salida: {slide_output_dir}")

        # Crear el directorio de salida principal
        os.makedirs(slide_output_dir, exist_ok=True)

        # Crear subdirectorios solo si son necesarios
        masks_output_dir = None
        if save_mask:
            masks_output_dir = slide_output_dir / "masks"
            os.makedirs(masks_output_dir, exist_ok=True)
            logger.info(f"Se guardarán máscaras en: {masks_output_dir}")

        original_output_dir = None
        if save_original:
            original_output_dir = slide_output_dir / "original"
            os.makedirs(original_output_dir, exist_ok=True)
            logger.info(f"Se guardarán imágenes originales en: {original_output_dir}")

        overlay_output_dir = None

        geojson_output_dir = None
        if save_geojson:
            count_output_dir = slide_output_dir / "count"
            os.makedirs(count_output_dir, exist_ok=True)
            logger.info(f"Se ejecutará conteo de glomérulos en: {count_output_dir}")

        save_level_image(slide, slide_output_dir, level=7, bounds_x=bounds_x, bounds_y=bounds_y, bounds_height=bounds_height, bounds_width=bounds_width, slide_name=slide_name)

        # ROI
        x_roi = bounds_x
        y_roi = bounds_y
        width_roi = bounds_width
        height_roi = bounds_height

        # ROI 1024x1024
        if save_roi:
            roi_output_path = slide_output_dir / f"{slide_name}_ROI_1024x1024.png"
            save_roi_image(slide, x_roi, y_roi, width_roi, height_roi, roi_output_path)

        downsample = slide.level_downsamples[level] if level > 0 else 1

        logger.info(f"Procesando {slide_name} (ROI: {width_roi}x{height_roi} en nivel {level})")
        logger.info(f"Filtros: extremos > {extreme_threshold*100}% | varianza < {min_variance}")
        logger.info(f"Filtro de fondo: activado={skip_background} | umbral={background_threshold} | píxeles extremos={extreme_pixel_ratio}")

        patch_id = 0
        valid_patches = 0
        rejected_patches = 0
        background_patches = 0  # Contador para patches de fondo ignorados

        masks_dict = {}
        original_dict = {}

        # Convertir coordenadas ROI al nivel especificado
        x_roi_level = int(x_roi / downsample) if level > 0 else x_roi
        y_roi_level = int(y_roi / downsample) if level > 0 else y_roi
        width_roi_level = int(width_roi / downsample) if level > 0 else width_roi
        height_roi_level = int(height_roi / downsample) if level > 0 else height_roi
        
        logger.info(f"ROI en nivel {level}: {x_roi_level}, {y_roi_level}, {width_roi_level}, {height_roi_level}")

        # Definir el tamaño del paso (stride) para los parches - asegurar cobertura completa
        stride = patch_size - overlap  # Asumimos solapamiento del 50%

        # Calcular el número total de parches para la barra de progreso
        total_patches = ((width_roi_level // stride) + 1) * ((height_roi_level // stride) + 1)
        
        # Log para depuración
        logger.info(f"Procesando aproximadamente {total_patches} parches con stride={stride}")

        # Asegurar que procesamos hasta el final de la imagen
        # Ajustamos los límites para incluir parches completos hasta el final 
        x_max = x_roi_level + width_roi_level
        y_max = y_roi_level + height_roi_level
        
        # Variables para el cálculo del tiempo estimado
        start_time = time.time()
        last_time_update = start_time
        processing_times = []  # Almacenar tiempos de procesamiento para cada parche
        
        # Iterar sobre todos los parches con el stride definido
        for y in range(y_roi_level, y_max, stride):
            for x in range(x_roi_level, x_max, stride):
                # Marca de tiempo para este parche
                patch_start_time = time.time()
                
                # Calcular dimensiones reales del parche (pueden ser menores en los bordes)
                actual_width = min(patch_size, x_roi_level + width_roi_level - x)
                actual_height = min(patch_size, y_roi_level + height_roi_level - y)
                
                # Procesar todos los parches, incluso los del borde
                # Convertir coordenadas del nivel actual a nivel 0 (resolución completa)
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                
                # Leer el parche
                try:
                    # Si el parche es menor que el tamaño completo, es un parche de borde
                    is_edge_patch = actual_width < patch_size or actual_height < patch_size
                    
                    if is_edge_patch:
                        # Para parches de borde, creamos un parche del tamaño estándar con fondo blanco
                        # y colocamos el contenido real en la esquina superior izquierda
                        patch = slide.read_region((x0, y0), level, (actual_width, actual_height))
                        patch = patch.convert("RGB")
                        
                        # Crear un lienzo del tamaño completo (con fondo blanco)
                        full_patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
                        
                        # Colocar el parche real en la esquina superior izquierda
                        full_patch.paste(patch, (0, 0))
                        
                        # Convertir a numpy array
                        patch_array = np.array(full_patch)
                    else:
                        # Para parches completos, simplemente leerlos directamente
                        patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                        patch = patch.convert("RGB")
                        patch_array = np.array(patch)

                    # Calcular coordenadas relativas al ROI (importantes para posicionamiento)
                    x_rel = x0 - bounds_x
                    y_rel = y0 - bounds_y
                    
                    # Comprobar si es un parche de fondo
                    if skip_background and is_background_patch(patch_array, background_threshold, extreme_pixel_ratio):
                        background_patches += 1
                        if background_patches % 100 == 0:  # Solo loguear cada 100 parches de fondo
                            logger.info(f"Ignorados {background_patches} parches de fondo hasta ahora")
                        continue  # Saltar este parche y continuar con el siguiente

                    # Guardar imagen original si se solicita
                    

                    # Guardar en diccionario para referencia posterior
                    # Importante: Usamos las coordenadas relativas al ROI para posicionamiento correcto
                    original_dict[(x_rel, y_rel)] = patch_array

                    # Convertir a BGR para el modelo
                    patch_bgr = cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR)

                    # Ejecutar inferencia del modelo
                    result = inference_model(model, patch_bgr)
                    glomeruli_mask = get_glomeruli_mask(result, patch_size, patch_size)

                    # Restringir la máscara si es un parche de borde
                    if is_edge_patch:
                        # Si es un parche de borde, asegurar que la máscara solo tenga contenido en la zona válida
                        edge_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                        edge_mask[0:actual_height, 0:actual_width] = glomeruli_mask[0:actual_height, 0:actual_width]
                        glomeruli_mask = edge_mask

                    # Guardar máscara si se solicita
                    if save_mask and masks_output_dir:                        
                        mask_filename = f"{slide_name}_mask{patch_id:04d}_x{x_rel}_y{y_rel}.png"
                        cv2.imwrite(str(masks_output_dir / mask_filename), glomeruli_mask)

                    if save_original and original_output_dir:
                        overlay_array = create_overlay_image(patch_array, glomeruli_mask, alpha=overlay_alpha, color_name=mask_color)

                        # Añadir las líneas de overlap
                        overlay_array_with_lines = add_overlap_lines_to_array(overlay_array, overlap=overlap)

                        orig_filename = f"{slide_name}_patch{patch_id:04d}_x{x_rel}_y{y_rel}.png"
                        Image.fromarray(overlay_array_with_lines).save(original_output_dir / orig_filename)
                        
                    # Guardar máscara en diccionario con coordenadas relativas
                    masks_dict[(x_rel, y_rel)] = glomeruli_mask
                    
                    # Registrar el tiempo que tomó procesar este parche
                    patch_end_time = time.time()
                    patch_time = patch_end_time - patch_start_time
                    processing_times.append(patch_time)
                    
                    # Actualizar y mostrar el tiempo estimado cada 50 parches
                    if patch_id % 50 == 0 and patch_id > 0:
                        # Calcular tiempo promedio por parche usando los últimos 50 parches (o menos al inicio)
                        recent_times = processing_times[-50:]
                        avg_time_per_patch = sum(recent_times) / len(recent_times)
                        
                        # Estimar tiempo restante
                        remaining_patches = total_patches - patch_id - background_patches
                        estimated_time_remaining = remaining_patches * avg_time_per_patch
                        
                        # Formatear tiempo restante como HH:MM:SS
                        time_remaining_str = str(timedelta(seconds=int(estimated_time_remaining)))
                        
                        # Calcular porcentaje completado
                        percent_complete = ((patch_id + background_patches) / total_patches) * 100
                        
                        # Mostrar información de progreso
                        elapsed_time = time.time() - start_time
                        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
                        
                        logger.info(f"Progreso: {patch_id + background_patches}/{total_patches} parches ({percent_complete:.1f}%) | "
                                  f"Procesados: {patch_id} | Ignorados (fondo): {background_patches} | "
                                  f"Tiempo transcurrido: {elapsed_time_str} | "
                                  f"Tiempo restante estimado: {time_remaining_str} | "
                                  f"Fin estimado: {datetime.now() + timedelta(seconds=estimated_time_remaining)}")
                        
                        # Actualizar la marca de tiempo de la última actualización
                        last_time_update = time.time()

                    patch_id += 1
                    valid_patches += 1
                except Exception as e:
                    logger.error(f"Error en la predicción del parche en x={x}, y={y}: {e}")
                    rejected_patches += 1

        # Mostrar tiempo total de procesamiento
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        logger.info(f"Tiempo total de procesamiento: {total_time_str}")
        logger.info(f"Parches procesados: {valid_patches} | Parches de fondo ignorados: {background_patches} | Parches con error: {rejected_patches}")

        # Generar y guardar la imagen compuesta del tamaño exacto de la ROI
        if save_composite:
            logger.info("Generando imagen compuesta de la ROI completa...")
            composite_path = slide_output_dir / f"{slide_name}_glomeruli_composite.png"
            results_composite = save_composite_mask(masks_dict, original_dict, composite_path, 
                              roi_dimensions=(width_roi_level, height_roi_level),
                              alpha=overlay_alpha, color_name=mask_color, overlap=overlap)
            
        if save_geojson and masks_dict:
            count_output_dir = slide_output_dir / "count"
            os.makedirs(count_output_dir, exist_ok=True)
            logger.info(f"Ejecutando conteo de glomérulos en: {count_output_dir}")
            
            # Ejecutar conteo usando la imagen compuesta
            composite_path = slide_output_dir / f"{slide_name}_glomeruli_composite.png"
            
            if composite_path.exists():
                logger.info(f"Ejecutando calibración y conteo para: {composite_path}")
                
                # Crear instancia del calibrador y ejecutar proceso completo
                calibrador = GlomeruliParameterCalibrator()
                resultado_conteo = calibrador.process_complete(
                    image_path=str(composite_path),
                    output_dir=str(count_output_dir),
                    result=results_composite
                )
                geojson_path = resultado_conteo['results']['visualization']['geojson_path']
                if 'error' not in resultado_conteo:
                    logger.info("✅ Conteo completado exitosamente")
                    if 'visualization' in resultado_conteo['results']:
                        viz = resultado_conteo['results']['visualization']
                        logger.info(f"📊 Glomérulos detectados: {viz['detected_count']}")
                        logger.info(f"📁 Resultados en: {count_output_dir}")
                else:
                    logger.error(f"❌ Error en el conteo: {resultado_conteo['error']}")
            else:
                logger.warning(f"No se encontró la imagen compuesta: {composite_path}")
        white_zone = GlomeruliWhiteAreaAnalyzer(
            slide_path=slide_path,
            geojson_path=geojson_path,
            output_dir= slide_output_dir / "white",
            min_area_pixels=7,
            mask_expansion_pixels= 5,
        )
        
        # Procesar todos los glomérulos
        resultados = white_zone.process_all_glomeruli()
        logger.info(f"Procesamiento completo. Parches válidos: {valid_patches} | Ignorados (fondo): {background_patches} | Rechazados: {rejected_patches}")

    except Exception as e:
        logger.error(f"Error al procesar {slide_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'slide' in locals():
            slide.close()
            
        # Log de cierre
        logger.info(f"Procesamiento finalizado para {slide_name}")
        # Cerrar handlers de logging para liberar recursos
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
def generate_geojson(masks_dict, original_dict, slide_path, output_path, downsample, level, patch_size, overlap=512):
    """
    Genera archivos GeoJSON para QuPath con las máscaras detectadas:
    1. Un archivo conjunto con todas las máscaras individuales (sin fusionar/consolidar)
    2. Un archivo con máscaras consolidadas (fusionando glomérulos que aparecen en múltiples imágenes)
    
    Args:
        masks_dict: Diccionario con las coordenadas (x, y) como claves y las máscaras como valores
        original_dict: Diccionario con las coordenadas (x, y) como claves y las imágenes originales
        slide_path: Ruta al archivo MRXS original
        output_path: Ruta donde guardar el archivo GeoJSON
        downsample: Factor de escala del nivel procesado
        level: Nivel de la imagen procesada
        patch_size: Tamaño de los parches procesados
        overlap: Tamaño del solapamiento entre parches en píxeles (default: 512)
        
    Returns:
        tuple: Rutas a los archivos GeoJSON generados (conjunto sin fusionar, consolidado)
    """
    logger = get_logger()
    import cv2
    import json
    import uuid
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    
    if not masks_dict:
        logger.info("No hay máscaras para exportar a GeoJSON")
        return None, None
    
    # Nombre del archivo slide
    slide_name = Path(slide_path).stem
    
    # Crear la estructura GeoJSON para ambos archivos
    geojson_unconsolidated = {
        "type": "FeatureCollection",
        "features": []
    }
    
    geojson_consolidated = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Contar cuántas máscaras tenemos para procesar
    total_masks = len(masks_dict)
    logger.info(f"Procesando {total_masks} máscaras para exportar a GeoJSON...")
    logger.info(f"Usando solapamiento de {overlap} píxeles entre parches")
    
    # Determinar las dimensiones máximas para la máscara consolidada
    max_x = max(x for (x, y) in masks_dict.keys()) + patch_size
    max_y = max(y for (x, y) in masks_dict.keys()) + patch_size
    
    # Convertimos a coordenadas absolutas
    max_x_abs = int(max_x * downsample)
    max_y_abs = int(max_y * downsample)
    
    # Creamos una imagen en blanco para la máscara consolidada
    consolidated_mask = np.zeros((max_y_abs, max_x_abs), dtype=np.float32)
    
    # Crear una matriz para contar las contribuciones (para debugging)
    contribution_count = np.zeros((max_y_abs, max_x_abs), dtype=np.int32)
    
    # 1. Procesar máscaras individuales (sin consolidar)
    individual_roi_count = 0
    
    for (x, y), mask in masks_dict.items():
        # Convertir la máscara binaria a formato uint8
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0).astype(np.uint8) * 255
        else:
            binary_mask = mask.copy()
        
        # Calcular las coordenadas absolutas para el parche
        x_abs = int(x * downsample)
        y_abs = int(y * downsample)
        
        # Obtener contornos de la máscara individual
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filtrar contornos muy pequeños (posible ruido)
            if cv2.contourArea(contour) < 100:
                continue
            
            # Simplificar el contorno
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convertir contorno a formato GeoJSON (polígono)
            # Ajustar las coordenadas al offset del parche
            coordinates = []
            for point in approx:
                px, py = point[0]
                # Ajustar las coordenadas con el offset absoluto del parche
                coordinates.append([float(px + x_abs), float(py + y_abs)])
            
            # Para cerrar el polígono, el último punto debe ser igual al primero
            if coordinates and coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            
            # Si el contorno es válido, agregar a la lista de features
            if len(coordinates) >= 4:  # Un polígono cerrado necesita al menos 4 puntos
                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates]  # GeoJSON requiere un array anidado para polígonos
                    },
                    "properties": {
                        "classification": {
                            "name": "Positive",
                            "color": [0, 0, 255]  # Color en formato RGB [R, G, B]
                        },
                        "isLocked": False,
                        "measurements": []
                    }
                }
                
                geojson_unconsolidated["features"].append(feature)
                individual_roi_count += 1
    
    # 2. Contribuir a la máscara consolidada con fusión inteligente de solapamientos
    overlap_abs = int(overlap * downsample)  # Solapamiento en coordenadas absolutas
    half_overlap = overlap_abs // 2  # Mitad del solapamiento
    
    logger.info(f"Solapamiento absoluto: {overlap_abs} píxeles, mitad: {half_overlap} píxeles")
    
    # Obtener todas las posiciones únicas de parches para determinar vecinos
    patch_positions = list(masks_dict.keys())
    
    for (x, y), mask in masks_dict.items():
        # Convertir la máscara binaria a formato float32 para cálculos
        if mask.dtype != np.uint8:
            binary_mask = (mask > 0).astype(np.float32)
        else:
            binary_mask = (mask > 0).astype(np.float32)
        
        # Calcular las coordenadas absolutas para el parche
        x_abs = int(x * downsample)
        y_abs = int(y * downsample)
        
        h, w = binary_mask.shape
        
        # Asegurarnos de que no nos salimos de los límites
        end_x = min(x_abs + w, max_x_abs)
        end_y = min(y_abs + h, max_y_abs)
        
        # Ajustar el tamaño si es necesario
        mask_h = end_y - y_abs
        mask_w = end_x - x_abs
        
        if mask_h != h or mask_w != w:
            binary_mask = binary_mask[:mask_h, :mask_w]
            h, w = binary_mask.shape
        
        # Crear una máscara de contribución para este parche
        contribution_mask = np.ones_like(binary_mask, dtype=np.float32)
        
        # Buscar parches vecinos y ajustar la contribución en las zonas de solapamiento
        for (other_x, other_y) in patch_positions:
            if (other_x, other_y) == (x, y):
                continue
            
            other_x_abs = int(other_x * downsample)
            other_y_abs = int(other_y * downsample)
            
            # Verificar solapamiento horizontal (parche a la derecha)
            if (other_x > x and other_y == y and 
                x_abs + w > other_x_abs and other_x_abs < x_abs + w):
                
                # Hay solapamiento horizontal con parche de la derecha
                overlap_start = other_x_abs - x_abs
                overlap_end = min(w, overlap_start + overlap_abs)
                
                if overlap_start >= 0 and overlap_start < w:
                    # En la zona de solapamiento, solo contribuir con la mitad izquierda
                    mid_point = overlap_start + half_overlap
                    if mid_point < w:
                        contribution_mask[:, mid_point:overlap_end] = 0.0
            
            # Verificar solapamiento horizontal (parche a la izquierda)
            if (other_x < x and other_y == y and 
                other_x_abs + patch_size > x_abs):
                
                # Hay solapamiento horizontal con parche de la izquierda
                overlap_end = (other_x_abs + patch_size) - x_abs
                overlap_start = max(0, overlap_end - overlap_abs)
                
                if overlap_end > 0 and overlap_start < w:
                    # En la zona de solapamiento, solo contribuir con la mitad derecha
                    mid_point = overlap_start + half_overlap
                    if mid_point > 0:
                        contribution_mask[:, overlap_start:mid_point] = 0.0
            
            # Verificar solapamiento vertical (parche abajo)
            if (other_y > y and other_x == x and 
                y_abs + h > other_y_abs and other_y_abs < y_abs + h):
                
                # Hay solapamiento vertical con parche de abajo
                overlap_start = other_y_abs - y_abs
                overlap_end = min(h, overlap_start + overlap_abs)
                
                if overlap_start >= 0 and overlap_start < h:
                    # En la zona de solapamiento, solo contribuir con la mitad superior
                    mid_point = overlap_start + half_overlap
                    if mid_point < h:
                        contribution_mask[mid_point:overlap_end, :] = 0.0
            
            # Verificar solapamiento vertical (parche arriba)
            if (other_y < y and other_x == x and 
                other_y_abs + patch_size > y_abs):
                
                # Hay solapamiento vertical con parche de arriba
                overlap_end = (other_y_abs + patch_size) - y_abs
                overlap_start = max(0, overlap_end - overlap_abs)
                
                if overlap_end > 0 and overlap_start < h:
                    # En la zona de solapamiento, solo contribuir con la mitad inferior
                    mid_point = overlap_start + half_overlap
                    if mid_point > 0:
                        contribution_mask[overlap_start:mid_point, :] = 0.0
        
        # Aplicar la máscara de contribución
        final_contribution = binary_mask * contribution_mask
        
        # Agregar la contribución a la máscara consolidada
        consolidated_mask[y_abs:end_y, x_abs:end_x] += final_contribution
        
        # Actualizar contador de contribuciones (para debugging)
        contribution_count[y_abs:end_y, x_abs:end_x] += (contribution_mask > 0).astype(np.int32)
    
    # Convertir la máscara consolidada de vuelta a binaria
    consolidated_mask_binary = (consolidated_mask > 0.5).astype(np.uint8) * 255
    
    # 3. Encontrar contornos en la máscara consolidada
    consolidated_contours, _ = cv2.findContours(consolidated_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contador para ROIs consolidados
    consolidated_roi_count = 0
    
    # 4. Procesar cada contorno encontrado en la máscara consolidada
    for contour in consolidated_contours:
        # Filtrar contornos muy pequeños (posible ruido)
        if cv2.contourArea(contour) < 100:  # Ajustar este valor según sea necesario
            continue
        
        # Simplificar el contorno para reducir el número de puntos
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convertir contorno a formato GeoJSON (polígono)
        coordinates = []
        for point in approx:
            # Las coordenadas ya están en absolutas
            px, py = point[0]
            coordinates.append([float(px), float(py)])
        
        # Para cerrar el polígono, el último punto debe ser igual al primero
        if coordinates and coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        
        # Si el contorno es válido, agregar a la lista de features
        if len(coordinates) >= 4:  # Un polígono cerrado necesita al menos 4 puntos
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]  # GeoJSON requiere un array anidado para polígonos
                },
                "properties": {
                    "classification": {
                        "name": "Positive",
                        "color": [0, 0, 255]  # Color en formato RGB [R, G, B]
                    },
                    "isLocked": False,
                    "measurements": []
                }
            }
            
            geojson_consolidated["features"].append(feature)
            consolidated_roi_count += 1
    
    # Generar las rutas de salida para ambos archivos
    base_output_path = Path(str(output_path).replace('.qpdata', ''))
    unconsolidated_output_path = f"{base_output_path}_unconsolidated.geojson"
    consolidated_output_path = f"{base_output_path}_consolidated.geojson"
    
    # Escribir el archivo GeoJSON de máscaras individuales (conjunto sin fusionar)
    with open(unconsolidated_output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_unconsolidated, f, indent=2)
    
    # Escribir el archivo GeoJSON consolidado
    with open(consolidated_output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_consolidated, f, indent=2)
    
    logger.info(f"Exportación a GeoJSON completada:")
    logger.info(f"- {individual_roi_count} ROIs sin fusionar guardados en {unconsolidated_output_path}")
    logger.info(f"- {consolidated_roi_count} ROIs consolidados guardados en {consolidated_output_path}")
    
    
    return unconsolidated_output_path, consolidated_output_path

def procesar_imagen():
    args = parse_args()
    
    # Inicializar el logger global una sola vez
    setup_logger(args.input_dir)
    logger = get_logger()
    
    logger.info("Iniciando procesamiento con los siguientes parámetros:")
    logger.info(f"Tamaño de parche: {args.patch_size}")
    logger.info(f"Overlap: {args.overlap}")
    logger.info(f"Nivel: {args.level}")
    logger.info(f"Filtro de fondo: {args.skip_background} (umbral: {args.background_threshold}, píxeles extremos: {args.extreme_pixel_ratio})")
    
    input_path = Path(args.input_dir)
    
    # Inicializar el modelo
    logger.info("Inicializando modelo Mask2Former...")
    model = initialize_model(args.config_path, args.ckpt_path)
    logger.info("Modelo cargado correctamente")

    # Buscar archivos MRXS
    if input_path.is_file() and input_path.suffix.lower() == ".mrxs":
        mrxs_files = [str(input_path)]
    elif input_path.is_dir():
        mrxs_files = glob.glob(os.path.join(args.input_dir, "*.mrxs"))
    else:
        logger.error(f"❌ Ruta inválida: {args.input_dir}")
        return

    if not mrxs_files:
        logger.warning(f"❌ No se encontraron archivos MRXS en {args.input_dir}")
        return

    logger.info(f"🔍 Encontrados {len(mrxs_files)} archivos MRXS")

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
            background_threshold=args.background_threshold,
            extreme_pixel_ratio=args.extreme_pixel_ratio,
            save_roi=args.save_roi,
            save_original=args.save_original,
            save_mask=args.save_mask,
            save_composite=args.save_composite,
            save_geojson=args.save_geojson,
            overlay_alpha=args.overlay_alpha,
            mask_color=args.mask_color,
            overlap = args.overlap
        )

    logger.info("\n✅ Procesamiento completado")

if __name__ == "__main__":
    procesar_imagen()
    
    
    
    
    #TODO Cambiar el slide y el overlap y la metodologia de reconstuccion, tanto a nivel de mascara como de geojson. 