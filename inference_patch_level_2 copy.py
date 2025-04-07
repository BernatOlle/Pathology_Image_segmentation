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

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="Puede ser una sola imagen o un directorio de imágenes de una WSI")
parser.add_argument("--config", type=str, help="Ruta del archivo de configuración")
parser.add_argument("--ckpt", type=str, help="Ruta del checkpoint del modelo")
parser.add_argument("--stitch", action="store_true", help="Aplicar estrategia de stitching o no")
parser.add_argument("--img_size", type=int, help="2048 (KPIs) o 1024 (Mice glomeruli)")

def count_pixels(mask_path):
    """Cuenta los píxeles de cada valor en la máscara."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise Exception(f"No se pudo cargar la máscara: {mask_path}")

    unique, counts = np.unique(mask, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    
    print(f"Distribución de píxeles para {mask_path}: {pixel_counts}")
    return pixel_counts

def get_mask_path(img_path: str):
    """Obtiene la ruta de la máscara correspondiente a la imagen y cuenta sus píxeles."""
    # En caso de Mice glomeruli (orbit)
    mask_path = img_path.replace('/img/', '/mask/').replace('_img.jpg', '_mask.png')
    
    if os.path.isfile(mask_path):
        return mask_path
    
    # En caso de KPIs
    mask_path = mask_path.replace('_mask.png', '_mask.jpg')
    if os.path.isfile(mask_path):
        return mask_path
    
    raise Exception(f'No se encontró la máscara para {img_path}')


if __name__=="__main__":
    args = parser.parse_args()
    print(args)

    # define test_pipeline
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]

    # load model
    model = init_model(args.config, args.ckpt)
    model.cfg.test_pipeline = test_pipeline
    print(model.cfg.model.backbone.type)

    # get image paths
    if os.path.isdir(args.input):
        input_dir = Path(args.input)
        all_img_paths = glob.glob(str(Path(input_dir)/'**/*_img.*'), recursive=True)
    elif os.path.isfile(args.input):
        all_img_paths = [args.input]
    
    # check if stitching strategy can be performed
    is_stitching = args.stitch
    if is_stitching and len(all_img_paths) == 1:
        print(f'Encontrada solo 1 imagen, no se puede realizar stitching.')
        is_stitching = False

    print(f'Número de imágenes de entrada: {len(all_img_paths)}')
    all_wsi_ids, all_coords = utils.get_wsi_data(all_img_paths, args.img_size)

    if len(set(all_wsi_ids)) > 1:
        print(f'Las imágenes en {args.input} no son del mismo WSI, no se puede realizar stitching.')
        is_stitching = False

    mDice = 0.0

    if is_stitching:
        print(f'Realizando estrategia de stitching')
        all_coords = np.array(all_coords)

        max_x = np.max(all_coords[:, 2])
        max_y = np.max(all_coords[:, 3])

        min_size = args.img_size
        if max_x < min_size:
            max_x = min_size
        if max_y < min_size:
            max_y = min_size

        wsi_shape = [2, max_y, max_x]
        pred_wsi_data = torch.full(wsi_shape, 0, dtype=torch.float)

        pbar = tqdm(list(zip(all_img_paths, all_coords)), leave=True)
        for img_path, coord in pbar:
            img_data = cv2.imread(img_path, -1)
            x_min, y_min, x_max, y_max = coord

            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data
            raw_logits = torch.softmax(raw_logits, dim=0)
            raw_logits = raw_logits.cpu()

            pred_wsi_data[:, y_min:y_max, x_min:x_max] += raw_logits

        pbar = tqdm(list(zip(all_img_paths, all_coords)), leave=True)
        print("Recortando de nuevo: ")
        for img_path, coord in pbar:
            mask_path = get_mask_path(img_path)
            mask_data = cv2.imread(mask_path, -1)
            
            x_min, y_min, x_max, y_max = coord
            crop_pred_raw = pred_wsi_data[:, y_min:y_max, x_min:x_max]

            crop_pred_raw = torch.softmax(crop_pred_raw, dim=0)

            pred_max_value, pred_seg = crop_pred_raw.max(axis=0, keepdims=True)
            pred_seg = pred_seg.cpu().numpy()[0]

            # Convertir predicción en binario (0 y 255)
            binary_mask = np.where(pred_seg > 0, 255, 0).astype(np.uint8)

            # Guardar la máscara predicha
            output_folder = "outputs"
            os.makedirs(output_folder, exist_ok=True)
            save_path = os.path.join(output_folder, Path(img_path).stem + "_pred_mask.png")
            cv2.imwrite(save_path, binary_mask)

            # 🔹 Cargar imagen original y hacer la superposición con la máscara predicha
            original_image = cv2.imread(img_path)  # Cargar imagen original
            binary_mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)  # Colorear la máscara
            
            alpha = 0.5  # Nivel de transparencia
            overlay = cv2.addWeighted(original_image, 1, binary_mask_colored, alpha, 0)  # Superponer
            
            # Guardar la imagen superpuesta
            overlay_folder = os.path.join(output_folder, "overlay")
            os.makedirs(overlay_folder, exist_ok=True)
            overlay_path = os.path.join(overlay_folder, Path(img_path).stem + "_overlay.png")
            cv2.imwrite(overlay_path, overlay)

            # Calcular Dice Score
            dice_score = utils.calculate_dice(y_pred=pred_seg, y_gt=mask_data)
            mDice += dice_score
        
        print(f'Dice medio: {mDice/len(all_img_paths)}')

    else:
        print(f'Realizando segmentación por parche de imagen')
        for img_path in tqdm(all_img_paths):
            img_data = cv2.imread(img_path, -1)
            mask_path = get_mask_path(img_path)
            mask_data = cv2.imread(mask_path, -1)
            
            pred_res = inference_model(model, img_data)
            raw_logits = pred_res.seg_logits.data

            _, pred_seg = raw_logits.max(axis=0, keepdims=True)
            pred_seg = pred_seg.cpu().numpy()[0]

            dice_score = utils.calculate_dice(y_pred=pred_seg, y_gt=mask_data)
            mDice += dice_score

        print(f'Dice medio: {mDice/len(all_img_paths)}')
