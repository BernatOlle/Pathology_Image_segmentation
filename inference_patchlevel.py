import os
import glob
from pathlib import Path
import cv2
import numpy as np
import torch
from mmseg.apis import init_model, inference_model
import utils
from tqdm import tqdm
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="Directorio con los recortes de imágenes MRXS")
parser.add_argument("--config", type=str, help="Ruta del archivo de configuración")
parser.add_argument("--ckpt", type=str, help="Ruta del checkpoint del modelo")
parser.add_argument("--stitch", action="store_true", help="Aplicar estrategia de stitching o no")
parser.add_argument("--output", type=str, default="outputs", help="Directorio para guardar los resultados")
parser.add_argument("--mask_dir", type=str, help="Directorio con las máscaras de referencia (si existen)")
parser.add_argument("--scale_factor", type=float, default=0.6, help="Factor de escala para reducción de memoria (0-1)")

def extract_coordinates_from_filename(filename):
    """Extrae las coordenadas X e Y del nombre de archivo generado por el script de recortes."""
    match = re.search(r'_x(\d+)_y(\d+)', filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None, None

def get_slide_name_from_path(path):
    """Extrae el nombre de la lámina del path del recorte."""
    path_parts = Path(path).parts
    # Buscar el nombre de la lámina en la ruta
    for part in path_parts:
        if "_patch" not in part:
            slide_name = part.split('_patch')[0]
            return slide_name
    # Si no se encuentra, usar el nombre del directorio padre
    return Path(path).parent.name

def get_mask_path(img_path, mask_dir):
    """
    Intenta encontrar la máscara correspondiente si existe.
    Si no existe, devuelve None.
    """
    if not mask_dir:
        return None

    # Extraer el nombre base del archivo
    img_filename = Path(img_path).name
    mask_filename = img_filename.replace('.png', '_mask.png')

    # Buscar en el directorio de máscaras
    potential_mask_path = Path(mask_dir) / mask_filename

    if os.path.isfile(potential_mask_path):
        return str(potential_mask_path)

    return None

def get_patch_data(patch_paths):
    """
    Extrae información de los parches: ID de lámina, coordenadas y dimensiones.
    """
    patch_data = []
    slide_ids = []

    for path in patch_paths:
        filename = Path(path).name
        slide_name = get_slide_name_from_path(path)
        x, y = extract_coordinates_from_filename(filename)

        if x is not None and y is not None:
            # Cargar la imagen para obtener las dimensiones
            img = cv2.imread(path)
            if img is not None:
                height, width = img.shape[:2]
                patch_data.append((path, slide_name, x, y, x + width, y + height))
                slide_ids.append(slide_name)
            else:
                print(f"No se pudo cargar la imagen: {path}")

    return patch_data, slide_ids

def process_with_downsampling(slide_patches, model, scale_factor, output_dir, overlay_dir, mask_dir=None):
    """Procesa la diapositiva con factor de submuestreo para reducir memoria."""
    slide_name = slide_patches[0][1]
    print(f"Procesando lámina: {slide_name} con factor de escala: {scale_factor}")
    
    # Determinar dimensiones originales
    max_x_orig = max([p[4] for p in slide_patches])
    max_y_orig = max([p[5] for p in slide_patches])
    
    # Aplicar factor de escala
    max_x = int(max_x_orig * scale_factor)
    max_y = int(max_y_orig * scale_factor)
    
    print(f"Dimensiones originales: {max_x_orig}x{max_y_orig}")
    print(f"Dimensiones reducidas: {max_x}x{max_y} (factor: {scale_factor})")
    
    # Inicializar tensor para la lámina reducida
    wsi_shape = [2, max_y, max_x]
    pred_wsi_data = torch.zeros(wsi_shape, dtype=torch.float)
    
    # Generar predicciones para cada parche con coordenadas escaladas
    for patch_info in tqdm(slide_patches, desc=f"Generando predicciones para {slide_name}"):
        img_path, _, x_min_orig, y_min_orig, x_max_orig, y_max_orig = patch_info
        
        # Escalar coordenadas
        x_min = int(x_min_orig * scale_factor)
        y_min = int(y_min_orig * scale_factor)
        x_max = int(x_max_orig * scale_factor)
        y_max = int(y_max_orig * scale_factor)
        
        # Verificar dimensiones válidas
        if x_max <= x_min or y_max <= y_min:
            continue
            
        # Cargar y redimensionar imagen
        img_data = cv2.imread(img_path)
        width_scaled = x_max - x_min
        height_scaled = y_max - y_min
        
        if width_scaled <= 0 or height_scaled <= 0:
            continue
            
        scaled_img = cv2.resize(img_data, (width_scaled, height_scaled))
        
        pred_res = inference_model(model, scaled_img)
        raw_logits = pred_res.seg_logits.data
        raw_logits = torch.softmax(raw_logits, dim=0)
        raw_logits = raw_logits.cpu()
        
        # Verificar que las dimensiones coincidan
        if raw_logits.shape[1:] != (height_scaled, width_scaled):
            print(f"Advertencia: Dimensiones de logits ({raw_logits.shape[1:]}) no coinciden con imagen escalada ({height_scaled}, {width_scaled})")
            raw_logits = torch.nn.functional.interpolate(
                raw_logits.unsqueeze(0), 
                size=(height_scaled, width_scaled), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Agregar predicciones al tensor de la lámina
        try:
            pred_wsi_data[:, y_min:y_max, x_min:x_max] += raw_logits
        except Exception as e:
            print(f"Error al agregar predicciones: {e}")
            print(f"Shape de raw_logits: {raw_logits.shape}, region: {y_min}:{y_max}, {x_min}:{x_max}")
    
    # Evaluar cada parche y generar resultados
    mDice = 0.0
    total_patches_with_masks = 0
    
    for patch_info in tqdm(slide_patches, desc=f"Evaluando parches de {slide_name}"):
        img_path, _, x_min_orig, y_min_orig, x_max_orig, y_max_orig = patch_info
        
        # Redimensionar a coordenadas reducidas
        x_min = int(x_min_orig * scale_factor)
        y_min = int(y_min_orig * scale_factor)
        x_max = int(x_max_orig * scale_factor)
        y_max = int(y_max_orig * scale_factor)
        
        # Verificar dimensiones válidas
        if x_max <= x_min or y_max <= y_min:
            continue
            
        # Obtener predicción para este parche en coordenadas reducidas
        try:
            crop_pred_raw = pred_wsi_data[:, y_min:y_max, x_min:x_max]
            crop_pred_raw = torch.softmax(crop_pred_raw, dim=0)
            
            pred_max_value, pred_seg = crop_pred_raw.max(dim=0)
            pred_seg = pred_seg.cpu().numpy()
            
            # Escalar la máscara de predicción de vuelta al tamaño original
            height_orig = y_max_orig - y_min_orig
            width_orig = x_max_orig - x_min_orig
            
            pred_seg_resized = cv2.resize(
                pred_seg.astype(np.uint8), 
                (width_orig, height_orig), 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Convertir predicción en binario (0 y 255)
            binary_mask = np.where(pred_seg_resized > 0, 255, 0).astype(np.uint8)
            
            # Guardar la máscara predicha
            patch_filename = Path(img_path).name
            save_path = output_dir / f"{Path(img_path).stem}_pred_mask.png"
            cv2.imwrite(str(save_path), binary_mask)
            
            # Crear overlay con la imagen original
            original_image = cv2.imread(img_path)
            binary_mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
            
            alpha = 0.5
            overlay = cv2.addWeighted(original_image, 1, binary_mask_colored, alpha, 0)
            
            # Guardar overlay
            overlay_path = overlay_dir / f"{Path(img_path).stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            
            # Calcular Dice Score si hay máscara disponible
            mask_path = get_mask_path(img_path, mask_dir)
            if mask_path:
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Normalizar máscara a valores binarios (0 y 1) para calcular el Dice
                mask_data = (mask_data > 0).astype(np.uint8)
                pred_normalized = (pred_seg_resized > 0).astype(np.uint8)
                
                dice_score = utils.calculate_dice(y_pred=pred_normalized, y_gt=mask_data)
                mDice += dice_score
                total_patches_with_masks += 1
        
        except Exception as e:
            print(f"Error al procesar parche {img_path}: {e}")
    
    return mDice, total_patches_with_masks

def process_individual_patches(patch_paths, model, output_dir, overlay_dir, mask_dir=None):
    """Procesa cada parche individualmente sin stitching."""
    mDice = 0.0
    total_patches_with_masks = 0
    
    for img_path in tqdm(patch_paths, desc="Procesando parches individuales"):
        try:
            img_data = cv2.imread(img_path)
            
            # Obtener predicción
            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data
            
            _, pred_seg = raw_logits.max(dim=0)
            pred_seg = pred_seg.cpu().numpy()
            
            # Convertir predicción en binario (0 y 255)
            binary_mask = np.where(pred_seg > 0, 255, 0).astype(np.uint8)
            
            # Guardar la máscara predicha
            save_path = output_dir / f"{Path(img_path).stem}_pred_mask.png"
            cv2.imwrite(str(save_path), binary_mask)
            
            # Crear overlay con la imagen original
            binary_mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_data, 1, binary_mask_colored, 0.5, 0)
            
            # Guardar overlay
            overlay_path = overlay_dir / f"{Path(img_path).stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            
            # Calcular Dice Score si hay máscara disponible
            mask_path = get_mask_path(img_path, mask_dir)
            if mask_path:
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Normalizar máscara a valores binarios (0 y 1) para calcular el Dice
                mask_data = (mask_data > 0).astype(np.uint8)
                pred_normalized = (pred_seg > 0).astype(np.uint8)
                
                dice_score = utils.calculate_dice(y_pred=pred_normalized, y_gt=mask_data)
                mDice += dice_score
                total_patches_with_masks += 1
        
        except Exception as e:
            print(f"Error al procesar parche individual {img_path}: {e}")
    
    return mDice, total_patches_with_masks

if __name__=="__main__":
    print("Iniciando inferencia por parche...")
    args = parser.parse_args()
    print(args)
    
    # Crear directorios de salida
    output_dir = Path(args.output)
    overlay_dir = output_dir / "overlay"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # define test_pipeline
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]
    
    # load model
    model = init_model(args.config, args.ckpt)
    model.cfg.test_pipeline = test_pipeline
    print(f"Modelo cargado: {model.cfg.model.backbone.type}")
    
    # get image paths
    input_dir = Path(args.input)
    all_patch_paths = glob.glob(str(input_dir / "**" / "*.png"), recursive=True)
    
    if not all_patch_paths:
        print(f"No se encontraron imágenes en {args.input}")
        exit(1)
    
    print(f'Número de parches encontrados: {len(all_patch_paths)}')
    
    # Extraer información de los parches
    patch_data, slide_ids = get_patch_data(all_patch_paths)
    unique_slides = set(slide_ids)
    
    print(f'Número de láminas encontradas: {len(unique_slides)}')
    
    # Verificar factor de escala
    scale_factor = args.scale_factor
    if scale_factor <= 0 or scale_factor > 1:
        print(f"Factor de escala {scale_factor} inválido. Usando valor predeterminado de 0.25")
        scale_factor = 0.25
        
    print(f"Usando factor de escala: {scale_factor}")
    
    # Comprobar si se aplica stitching
    is_stitching = args.stitch
    if is_stitching and len(all_patch_paths) == 1:
        print(f'Encontrado solo 1 parche, no se puede realizar stitching.')
        is_stitching = False
    
    # Total de métricas
    total_mDice = 0.0
    total_patches_with_masks = 0
    
    if is_stitching:
        print(f'Realizando estrategia de stitching con submuestreo')
        
        # Procesar cada lámina por separado
        for slide_name in unique_slides:
            # Filtrar parches de esta lámina
            slide_patches = [p for p in patch_data if p[1] == slide_name]
            
            # Procesar con submuestreo
            mDice, patches_with_masks = process_with_downsampling(
                slide_patches, 
                model, 
                scale_factor, 
                output_dir, 
                overlay_dir, 
                args.mask_dir
            )
            
            # Actualizar totales
            total_mDice += mDice
            total_patches_with_masks += patches_with_masks
    
    else:
        print(f'Realizando segmentación por parche individual')
        
        # Procesar cada parche individualmente
        mDice, patches_with_masks = process_individual_patches(
            all_patch_paths, 
            model, 
            output_dir, 
            overlay_dir, 
            args.mask_dir
        )
        
        # Actualizar totales
        total_mDice += mDice
        total_patches_with_masks += patches_with_masks
    
    # Mostrar resultados finales
    if total_patches_with_masks > 0:
        print(f'Dice medio: {total_mDice/total_patches_with_masks}')
    else:
        print('No se encontraron máscaras de referencia para calcular el Dice.')

    print("Proceso completado. Resultados guardados en:", args.output)