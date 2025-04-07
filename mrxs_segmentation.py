import os
import glob
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import openslide
from PIL import Image
import cv2
import shutil

# Importar funciones de mmseg para la segmentación
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extracción y segmentación integrada de parches de imágenes MRXS, con guardado individual de máscaras, parches tintados y combinación final."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Archivo MRXS o directorio con imágenes MRXS")
    parser.add_argument("--patch_size", type=int, default=2048,
                        help="Tamaño de los parches (default: 2048)")
    parser.add_argument("--stride", type=int, default=2048,
                        help="Stride para extracción de parches (default: igual a patch_size, sin solapamiento)")
    parser.add_argument("--level", type=int, default=0,
                        help="Nivel de zoom para procesar la imagen (default: 0)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Ruta al archivo de configuración del modelo SegFormer")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Ruta al checkpoint del modelo SegFormer")
    parser.add_argument("--scale_factor", type=int, default=4,
                        help="Factor de reducción para la imagen combinada (default: 4)")
    parser.add_argument("--margin", type=int, default=20,
                        help="Margen adicional para recortar la imagen combinada (default: 20)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directorio general para guardar los outputs (default: outputs)")
    return parser.parse_args()

def init_segmentation_model(config_path, ckpt_path):
    # Inicializa el contexto y carga el modelo de segmentación
    init_default_scope('mmseg')
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]
    model = init_model(config_path, ckpt_path)
    model.cfg.test_pipeline = test_pipeline
    return model

def is_almost_uniform(patch_array, threshold=0.95):
    """
    Determina si el parche es casi todo negro o casi todo blanco.
    Se considera casi uniforme si la proporción de píxeles negros (valores <=10)
    o blancos (valores >=245) supera el umbral indicado.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    # Calcular el histograma de la imagen en escala de grises
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total_pixels = gray.size
    # Proporción de píxeles negros y blancos
    black_ratio = hist[:11].sum() / total_pixels
    white_ratio = hist[245:].sum() / total_pixels
    return black_ratio > threshold or white_ratio > threshold

def combine_masks(image_data, scale_factor, margin, combined_mask_path, cropped_mask_path):
    """
    Combina las máscaras predichas de cada parche.
    image_data: lista de tuplas (x, y, mask) con la coordenada superior izquierda y la máscara (de tamaño patch_size).
    """
    if not image_data:
        print("No hay máscaras para combinar.")
        return
    # Determinar el tamaño total necesario en la escala original
    max_x = max(x + mask.shape[1] for x, y, mask in image_data)
    max_y = max(y + mask.shape[0] for x, y, mask in image_data)
    new_width = max_x // scale_factor
    new_height = max_y // scale_factor

    combined_mask = np.zeros((new_height, new_width), dtype=np.float32)

    for x, y, mask in image_data:
        # Reducir la resolución de la máscara según el scale_factor
        small_mask = cv2.resize(mask, (mask.shape[1] // scale_factor, mask.shape[0] // scale_factor),
                                interpolation=cv2.INTER_NEAREST)
        x_small, y_small = x // scale_factor, y // scale_factor
        combined_mask[y_small:y_small + small_mask.shape[0],
                      x_small:x_small + small_mask.shape[1]] += small_mask.astype(np.float32)

    # Normalizar la imagen combinada a rango 0-255
    if combined_mask.max() > 0:
        combined_mask = (combined_mask / combined_mask.max()) * 255
    combined_mask = combined_mask.astype(np.uint8)
    cv2.imwrite(str(combined_mask_path), combined_mask)

    # Recortar la zona con contenido
    coords = np.column_stack(np.where(combined_mask > 0))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_min = max(y_min - margin, 0)
        x_min = max(x_min - margin, 0)
        y_max = min(y_max + margin, combined_mask.shape[0])
        x_max = min(x_max + margin, combined_mask.shape[1])
        cropped_mask = combined_mask[y_min:y_max, x_min:x_max]
    else:
        cropped_mask = combined_mask
    cv2.imwrite(str(cropped_mask_path), cropped_mask)

def process_slide(slide_path, model, patch_size, stride, level, output_dir):
    """
    Procesa un archivo MRXS: extrae parches de tamaño patch_size (usando stride),
    aplica la segmentación en cada parche, guarda la máscara predicha y almacena (x, y, máscara) en una lista.
    Además, guarda los parches tintados.
    Se utiliza una barra de progreso para mostrar el avance.
    """
    try:
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level] if level > 0 else 1

        # Calcular posiciones válidas para parches completos
        valid_x = [x for x in range(0, width, stride) if width - x >= patch_size]
        valid_y = [y for y in range(0, height, stride) if height - y >= patch_size]
        total_patches = len(valid_x) * len(valid_y)

        pbar = tqdm(total=total_patches, desc=f"Procesando {slide_name}", unit="parche")

        image_data = []  # Lista para almacenar (x, y, mask)
        patch_id = 0

        # Directorios de salida para máscaras y parches tintados
        patch_masks_dir = Path(output_dir) / "patch_masks" / slide_name
        patch_masks_dir.mkdir(parents=True, exist_ok=True)
        tinted_patches_dir = Path(output_dir) / "tinted_patches" / slide_name
        tinted_patches_dir.mkdir(parents=True, exist_ok=True)

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                actual_width = min(patch_size, width - x)
                actual_height = min(patch_size, height - y)
                # Procesar solo parches completos
                if actual_width != patch_size or actual_height != patch_size:
                    continue

                # Extraer parche
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                patch = patch.convert("RGB")
                patch_array = np.array(patch)

                # Verificar si el parche es casi uniforme (todo negro o todo blanco)
                # Aunque se detecte uniformidad, el parche no se descarta y se procesa de igual forma.
                if is_almost_uniform(patch_array):
                    # Aquí podríamos marcar o registrar de alguna forma que el parche es casi uniforme
                    pass

                # Aplicar tinte
                alpha_r = 0.1  # Intensidad del rojo
                alpha_b = 0.8  # Intensidad del azul
                patch_array[:, :, 0] = np.clip(patch_array[:, :, 0] + alpha_r * 255, 0, 255)
                patch_array[:, :, 2] = np.clip(patch_array[:, :, 2] + alpha_b * 255, 0, 255)
                tinted_patch_filename = f"tinted_{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                tinted_patch_path = tinted_patches_dir / tinted_patch_filename
                tinted_patch_bgr = cv2.cvtColor(patch_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(tinted_patch_path), tinted_patch_bgr)

                # Inferencia de segmentación
                pred_res = inference_model(model, patch_array)
                raw_logits = pred_res.seg_logits.data
                _, pred_mask = raw_logits.max(axis=0, keepdims=True)
                pred_mask = pred_mask.cpu().numpy()[0]
                pred_mask = (pred_mask.astype(np.uint8)) * 255

                # Guardar máscara individual
                patch_mask_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                patch_mask_path = patch_masks_dir / patch_mask_filename
                cv2.imwrite(str(patch_mask_path), pred_mask)

                image_data.append((x, y, pred_mask))
                patch_id += 1
                pbar.update(1)

        pbar.close()
        return image_data

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
        return []
    finally:
        if 'slide' in locals():
            slide.close()

def main():
    args = parse_args()

    # Crear directorio general de outputs y subcarpetas
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Directorio para máscaras combinadas
    combined_masks_dir = output_dir / "combined_masks"
    combined_masks_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar el modelo de segmentación (una sola vez)
    model = init_segmentation_model(args.config_path, args.ckpt_path)

    # Generar lista de archivos MRXS a procesar
    input_path = Path(args.input_dir)
    if input_path.is_file() and input_path.suffix.lower() == ".mrxs":
        mrxs_files = [str(input_path)]
    elif input_path.is_dir():
        mrxs_files = glob.glob(os.path.join(args.input_dir, "*.mrxs"))
    else:
        print(f"❌ Error: La ruta '{args.input_dir}' no es un archivo MRXS ni un directorio válido.")
        return
    if not mrxs_files:
        print(f"❌ No se encontraron archivos MRXS en '{args.input_dir}'")
        return

    # Procesar cada slide y acumular las máscaras predichas
    all_image_data = []
    for slide_path in mrxs_files:
        slide_data = process_slide(slide_path, model, args.patch_size, args.stride, args.level, output_dir)
        all_image_data.extend(slide_data)

    # Definir rutas para las máscaras combinadas dentro del directorio de outputs
    combined_mask_path = combined_masks_dir / "combined_mask.png"
    cropped_mask_path = combined_masks_dir / "cropped_combined_mask.png"

    # Combinar las máscaras de todos los parches en una imagen global
    combine_masks(all_image_data, args.scale_factor, args.margin,
                  combined_mask_path, cropped_mask_path)

    print("✅ Proceso completo.")

if __name__ == "__main__":
    main()
