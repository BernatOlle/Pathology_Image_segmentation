import os
from pathlib import Path

from mmseg.apis import init_model, inference_model

from tqdm import tqdm

import cv2
import torch
import numpy as np
import tifffile
import scipy.ndimage

import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="path to WSI file")
parser.add_argument("--config", type=str, help="config path")
parser.add_argument("--ckpt", type=str, help="checkpoint path")
parser.add_argument("--patch_size", type=int, default=2048)
parser.add_argument("--stride", type=int, default=1024)

def get_wsi_mask_path(wsi_path: str):
    """Get wsi-level for KPIs data
    Data structure:
    WSI data: test/12-299_wsi.tiff
    WSI mask: test/12-299_mask.tiff
    """
    mask_path = wsi_path.replace('_wsi.tiff', '_mask.tiff')
    if os.path.isfile(mask_path):
        return mask_path
    else:
        raise Exception(f'No ground-truth mask found for {wsi_path}!')
    
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
    # assign test_pipeline
    model.cfg.test_pipeline = test_pipeline
    print(model.cfg.model.backbone.type)

    mean_Dice = 0.0

    wsi_path = args.input
    wsi_name = Path(wsi_path).stem

    # get gt mask data
    mask_path = get_wsi_mask_path(wsi_path)

    # data is already in RGB. No need to remove non-tissue area since WSIs are already processed
    wsi_data = tifffile.imread(wsi_path, key=0)
    H, W, _ = wsi_data.shape
    mask_data = tifffile.imread(mask_path, key=0)

    assert (H, W) == mask_data.shape, f'WSI and GT mask not the same shape. {(H, W)} != {mask_data.shape}'

    # make sure to i nference on 40X digital magnification
    if '/NEP25/' not in wsi_path:
        lv = 2 
        wsi_data = scipy.ndimage.zoom(wsi_data, (1/lv, 1/lv, 1), order=1)
        H, W, _ = wsi_data.shape
        mask_data = cv2.resize(mask_data, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

    # normalize mask data to [0, 1]
    mask_data = mask_data/mask_data.max()
    mask_data = mask_data.astype(np.uint8)

    # striding and predict
    wsi_shape = [2, H, W]

    x_slide = int((W - args.patch_size) / args.stride) + 1
    y_slide = int((H - args.patch_size) / args.stride) + 1

    # save predictions from all models
    pred_wsi_data = torch.full(wsi_shape, 0, dtype=torch.float)

    pbar = tqdm(range(x_slide*y_slide), leave=True)
    pbar.set_description(f'{wsi_name}')
    print("Start")
    for xi in range(x_slide):
        for yi in range(y_slide):
            # update progress bar
            pbar.update(1)

            if xi == x_slide - 1:
                x_min = W - args.patch_size
            else:
                x_min = xi * args.stride

            if yi == y_slide - 1:
                y_min = H - args.patch_size
            else:
                y_min = yi * args.stride

            sub_wsi = wsi_data[y_min:y_min + args.patch_size, x_min:x_min + args.patch_size, :]
            # convert RGB to BGR
            sub_wsi = cv2.cvtColor(sub_wsi, cv2.COLOR_RGB2BGR)

            assert sub_wsi.shape == (2048, 2048, 3), f'Wrong shape {sub_wsi.shape}'

            # skip if image is non-tissue (i.e., all black)
            if len(np.unique(sub_wsi))==1:
                continue
            
            # predict
            pred_result = inference_model(model, sub_wsi)
            raw_logits = pred_result.seg_logits.data
            # softmax
            raw_logits = torch.softmax(raw_logits, dim=0)
            raw_logits = raw_logits.cpu()

            # store raw predictions
            pred_wsi_data[:, y_min:y_min + args.patch_size, x_min:x_min + args.patch_size] += raw_logits

    # normalize with softmax
    pred_wsi_data = torch.softmax(pred_wsi_data, dim=0)
    
    # get the predicted mask
    _, pred_seg = pred_wsi_data.max(axis=0, keepdims=True)
    pred_seg = pred_seg.cpu().numpy()[0]
    pred_seg = pred_seg.astype(np.uint8)



    # Calcular DICE score
    dice_score = utils.calculate_dice(y_pred=binary_mask, y_gt=mask_data)
    print(f'{wsi_name} - Dice: {dice_score}')
   
       # Convertir predicción en binario (0 y 255)
    binary_mask = np.where(pred_seg > 0, 255, 0).astype(np.uint8)

    # Reducir resolución (ejemplo: 4 veces más pequeña)
    scale_factor = 0.25  # Reducir al 25% del tamaño original (ajústalo según necesites)
    new_W = int(W * scale_factor)
    new_H = int(H * scale_factor)

    resized_mask = cv2.resize(binary_mask, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

    # Crear carpeta de salida si no existe
    output_folder = "outputs_2"
    os.makedirs(output_folder, exist_ok=True)

    # Definir la ruta de guardado
    save_path = os.path.join(output_folder, f"{wsi_name}_pred_mask.png")

    # Guardar la máscara reducida en PNG
    cv2.imwrite(save_path, resized_mask)

    print(f"Máscara predicha guardada en: {save_path} con resolución reducida a {new_W}x{new_H}")
