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
parser.add_argument("--mask_dir", type=str, help="Directorio con las máscaras de referencia (si existen)")
parser.add_argument("--scale_factor", type=float, default=0.1, help="Factor de escala para reducción de memoria (0-1)")

def extract_coordinates_from_filename(filename):
    """Extrae las coordenadas X e Y del nombre de archivo generado por el script de recortes."""
    match = re.search(r'_x(\d+)_y(\d+)', filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None, None

def get_slide_name_from_path(path):
    """
    Extrae el nombre de la lámina del path del recorte.
    Si el archivo se encuentra dentro de una carpeta "patches", se asume que el nombre de la lámina
    es el nombre de la carpeta padre de "patches". En otro caso, si el nombre del archivo contiene "_patch",
    se toma la parte anterior; de lo contrario, se toma el nombre de la carpeta contenedora.
    """
    p = Path(path)
    if p.parent.name.lower() == "patches":
        return p.parent.parent.name
    elif "_patch" in p.name:
        return p.name.split("_patch")[0]
    else:
        return p.parent.name

def get_mask_path(img_path, mask_dir):
    """Intenta encontrar la máscara correspondiente si existe."""
    if not mask_dir:
        return None

    img_filename = Path(img_path).name
    mask_filename = img_filename.replace('.png', '_mask.png')
    potential_mask_path = Path(mask_dir) / mask_filename

    if os.path.isfile(potential_mask_path):
        return str(potential_mask_path)
    return None

def get_patch_data(patch_paths):
    """
    Extrae información de los parches: ruta, ID de lámina, coordenadas y dimensiones.
    """
    patch_data = []
    slide_ids = []

    for path in patch_paths:
        filename = Path(path).name
        slide_name = get_slide_name_from_path(path)
        x, y = extract_coordinates_from_filename(filename)

        if x is not None and y is not None:
            img = cv2.imread(path)
            if img is not None:
                height, width = img.shape[:2]
                patch_data.append((path, slide_name, x, y, x + width, y + height))
                slide_ids.append(slide_name)
            else:
                print(f"No se pudo cargar la imagen: {path}")

    return patch_data, slide_ids

def process_with_downsampling(slide_patches, model, scale_factor, output_dir, mask_dir=None):
    """Procesa la lámina completa con downsampling para reducir el uso de memoria."""
    slide_name = slide_patches[0][1]
    print(f"Procesando lámina: {slide_name} con factor de escala: {scale_factor}")
    
    # Directorios de salida: se crean en la carpeta actual, en "result/<slide_name>/"
    pred_mask_dir = Path.cwd() / "result" / slide_name / "pred_mask"
    combined_dir = Path.cwd() / "result" / slide_name / "combinadas"
    os.makedirs(pred_mask_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    
    # Determinar dimensiones originales
    max_x_orig = max([p[4] for p in slide_patches])
    max_y_orig = max([p[5] for p in slide_patches])
    
    max_x = int(max_x_orig * scale_factor)
    max_y = int(max_y_orig * scale_factor)
    
    wsi_shape = [2, max_y, max_x]
    pred_wsi_data = torch.zeros(wsi_shape, dtype=torch.float)
    
    for patch_info in tqdm(slide_patches, desc=f"Generando predicciones para {slide_name}"):
        img_path, _, x_min_orig, y_min_orig, x_max_orig, y_max_orig = patch_info
        
        x_min = int(x_min_orig * scale_factor)
        y_min = int(y_min_orig * scale_factor)
        x_max = int(x_max_orig * scale_factor)
        y_max = int(y_max_orig * scale_factor)
        
        if x_max <= x_min or y_max <= y_min:
            continue
            
        img_data = cv2.imread(img_path)
        width_scaled = x_max - x_min
        height_scaled = y_max - y_min
        
        if width_scaled <= 0 or height_scaled <= 0:
            continue
            
        scaled_img = cv2.resize(img_data, (width_scaled, height_scaled))
        
        pred_res = inference_model(model, scaled_img)
        raw_logits = pred_res.seg_logits.data
        raw_logits = torch.softmax(raw_logits, dim=0).cpu()
        
        if raw_logits.shape[1:] != (height_scaled, width_scaled):
            print(f"Advertencia: Dimensiones de logits ({raw_logits.shape[1:]}) no coinciden con imagen escalada ({height_scaled}, {width_scaled})")
            raw_logits = torch.nn.functional.interpolate(
                raw_logits.unsqueeze(0), 
                size=(height_scaled, width_scaled), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        try:
            pred_wsi_data[:, y_min:y_max, x_min:x_max] += raw_logits
        except Exception as e:
            print(f"Error al agregar predicciones: {e}")
            print(f"Shape de raw_logits: {raw_logits.shape}, region: {y_min}:{y_max}, {x_min}:{x_max}")
    
    mDice = 0.0
    total_patches_with_masks = 0
    
    for patch_info in tqdm(slide_patches, desc=f"Evaluando parches de {slide_name}"):
        img_path, _, x_min_orig, y_min_orig, x_max_orig, y_max_orig = patch_info
        
        x_min = int(x_min_orig * scale_factor)
        y_min = int(y_min_orig * scale_factor)
        x_max = int(x_max_orig * scale_factor)
        y_max = int(y_max_orig * scale_factor)
        
        if x_max <= x_min or y_max <= y_min:
            continue
            
        try:
            crop_pred_raw = pred_wsi_data[:, y_min:y_max, x_min:x_max]
            crop_pred_raw = torch.softmax(crop_pred_raw, dim=0)
            
            _, pred_seg = crop_pred_raw.max(dim=0)
            pred_seg = pred_seg.cpu().numpy()
            
            height_orig = y_max_orig - y_min_orig
            width_orig = x_max_orig - x_min_orig
            
            pred_seg_resized = cv2.resize(
                pred_seg.astype(np.uint8), 
                (width_orig, height_orig), 
                interpolation=cv2.INTER_NEAREST
            )
            
            binary_mask = np.where(pred_seg_resized > 0, 255, 0).astype(np.uint8)
            
            save_path = pred_mask_dir / f"{Path(img_path).stem}_pred_mask.png"
            cv2.imwrite(str(save_path), binary_mask)
            
            original_image = cv2.imread(img_path)
            binary_mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_image, 1, binary_mask_colored, 0.5, 0)
            
            overlay_path = combined_dir / f"{Path(img_path).stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            
            mask_path = get_mask_path(img_path, mask_dir)
            if mask_path:
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_data = (mask_data > 0).astype(np.uint8)
                pred_normalized = (pred_seg_resized > 0).astype(np.uint8)
                dice_score = utils.calculate_dice(y_pred=pred_normalized, y_gt=mask_data)
                mDice += dice_score
                total_patches_with_masks += 1
        
        except Exception as e:
            print(f"Error al procesar parche {img_path}: {e}")
    
    return mDice, total_patches_with_masks

def process_individual_patches(patch_paths, model, output_dir, mask_dir=None):
    """Procesa cada parche individualmente sin aplicar stitching."""
    mDice = 0.0
    total_patches_with_masks = 0
    
    for img_path in tqdm(patch_paths, desc="Procesando parches individuales"):
        try:
            img_data = cv2.imread(img_path)
            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data
            _, pred_seg = raw_logits.max(dim=0)
            pred_seg = pred_seg.cpu().numpy()
            
            binary_mask = np.where(pred_seg > 0, 255, 0).astype(np.uint8)
            
            slide_name = get_slide_name_from_path(img_path)
            base_slide_dir = output_dir / slide_name
            pred_mask_dir = base_slide_dir / "pred_mask"
            combined_dir = base_slide_dir / "combined"
            os.makedirs(pred_mask_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)
            
            save_path = pred_mask_dir / f"{Path(img_path).stem}_pred_mask.png"
            cv2.imwrite(str(save_path), binary_mask)
            
            binary_mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_data, 1, binary_mask_colored, 0.5, 0)
            
            overlay_path = combined_dir / f"{Path(img_path).stem}_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            
            mask_path = get_mask_path(img_path, mask_dir)
            if mask_path:
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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
    
    # Se define el directorio de salida como el directorio actual
    output_dir = Path.cwd()
    
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]
    
    model = init_model(args.config, args.ckpt)
    model.cfg.test_pipeline = test_pipeline
    print(f"Modelo cargado: {model.cfg.model.backbone.type}")
    
    input_dir = Path(args.input) 
    all_patch_paths = glob.glob(str(input_dir /"paches" / "**" / "*.png"), recursive=True)
    
    if not all_patch_paths:
        print(f"No se encontraron imágenes en {args.input}")
        exit(1)
    
    print(f'Número de parches encontrados: {len(all_patch_paths)}')
    
    patch_data, slide_ids = get_patch_data(all_patch_paths)
    unique_slides = set(slide_ids)
    
    print(f'Número de láminas encontradas: {len(unique_slides)}')
    
    scale_factor = args.scale_factor
    if scale_factor <= 0 or scale_factor > 1:
        print(f"Factor de escala {scale_factor} inválido. Usando valor predeterminado de 0.25")
        scale_factor = 0.25
        
    print(f"Usando factor de escala: {scale_factor}")
    
    total_mDice = 0.0
    total_patches_with_masks = 0
    
    if args.stitch:
        print('Realizando estrategia de stitching con submuestreo')
        for slide_name in unique_slides:
            slide_patches = [p for p in patch_data if p[1] == slide_name]
            mDice, patches_with_masks = process_with_downsampling(
                slide_patches, 
                okey, 
                scale_factor, 
                output_dir, 
                args.mask_dir
            )
            total_mDice += mDice
            total_patches_with_masks += patches_with_masks
    else:
        print('Realizando segmentación por parche individual')
        mDice, patches_with_masks = process_individual_patches(
            all_patch_paths, 
            model, 
            output_dir, 
            args.mask_dir
        )
        total_mDice += mDice
        total_patches_with_masks += patches_with_masks
    
    if total_patches_with_masks > 0:
        print(f'Dice medio: {total_mDice/total_patches_with_masks}')
    else:
        print('No se encontraron máscaras de referencia para calcular el Dice.')

    print("Proceso completado. Resultados guardados en las carpetas correspondientes dentro del directorio actual.")
