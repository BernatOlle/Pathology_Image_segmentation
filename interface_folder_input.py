#!/usr/bin/env python3
import os
import cv2
import numpy as np
import re
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope
from pathlib import Path
import argparse

# Inicializar el contexto de mmseg
init_default_scope('mmseg')

# Definir pipeline de test
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='PackSegInputs'),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segmentación, combinación de máscaras y conteo de glomérulos en imágenes de láminas."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directorio de entrada correspondiente a la lámina. Se espera que contenga las imágenes (png, jpg, jpeg). Ejemplo: /ruta/a/slide-2023-02-18T08-17-59-R3-S17/")
    parser.add_argument("--config", type=str, required=True,
                        help="Ruta del archivo de configuración del modelo.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Ruta del checkpoint del modelo.")
    parser.add_argument("--mask_dir", type=str, default="",
                        help="Directorio con las máscaras de referencia (si existen).")
    parser.add_argument("--scale_factor", type=float, default=4,
                        help="Factor de escala para combinar las máscaras (default: 4).")
    return parser.parse_args()

def get_slide_name_from_path(path):
    """
    Extrae el nombre de la lámina a partir del directorio de entrada.
    Se asume que el directorio de entrada es la lámina.
    """
    print(Path(path).name)
    return Path(path).name

def main():
    args = parse_args()

    # Inicializar el modelo y asignar pipeline de test
    model = init_model(args.config, args.ckpt)
    model.cfg.test_pipeline = test_pipeline
    print(f"Modelo cargado: {model.cfg.model.backbone.type}")

    # Extraer el nombre de la lámina a partir del directorio de entrada
    slide_name = get_slide_name_from_path(args.input_dir)
    
    # Definir directorios de salida
    pred_mask_dir = Path.cwd() / "result" / slide_name / "pred_mask"
    combined_dir = Path.cwd() / "result" / slide_name / "combinadas"
    os.makedirs(pred_mask_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    images_dir = Path.cwd() / "result" / slide_name / "paches" / "matched"
    
    # Obtener lista de imágenes en el directorio de entrada
    print(images_dir)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No se encontraron imágenes en el directorio de entrada.")
        exit(1)
    
    # Expresión regular para extraer coordenadas (si se usan en el nombre)
    pattern = re.compile(r'_x(\d+)_y(\d+)')
    
    # Inicializar variable para contar los glomérulos totales
    total_glomeruli_count = 0
    glomeruli_counts = {}

    # Procesar cada imagen: inferencia y guardar máscara predicha
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img_data = cv2.imread(img_path, -1)
        if img_data is None:
            print(f"Error al cargar imagen: {img_path}")
            continue
        pred_res = inference_model(model, img_data)
        raw_logits = pred_res.seg_logits.data
        _, pred_mask = raw_logits.max(axis=0, keepdims=True)
        pred_mask = pred_mask.cpu().numpy()[0]
        # Se escala la máscara a rango 0-255
        mask_filename = pred_mask_dir / f'pred_{img_file}'
        cv2.imwrite(str(mask_filename), (pred_mask.astype(np.uint8) * 255))

        # =====================
        # CONTAR LOS GLOMERULOS EN CADA IMAGEN
        # =====================
        # Asegurarnos de que pred_mask esté en el formato adecuado y en escala de grises (un solo canal)
        if pred_mask.dtype != np.uint8:
            pred_mask = np.uint8(pred_mask)

        # Mejorar la binarización
        # Aplicamos un umbral adaptativo en lugar de un umbral fijo
        binary_mask = cv2.adaptiveThreshold(pred_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
        
        # Si los glomérulos no se detectan con el umbral adaptativo, intentar con un umbral fijo
        if np.count_nonzero(binary_mask) == 0:  # Si no hay glomérulos detectados, usar umbral fijo
            ret, binary_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)

        # Usar componentes conectados para contar las regiones (se asume que el fondo es la etiqueta 0)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        glomeruli_count = num_labels - 1  # Se descarta el fondo
        glomeruli_counts[img_file] = glomeruli_count
        total_glomeruli_count += glomeruli_count

    print("Predicción completada para todas las imágenes.")

    # =====================
    # COMBINAR LAS MÁSCARAS
    # =====================
    mask_files = [f for f in os.listdir(pred_mask_dir) if f.lower().endswith('.png')]
    image_data = []
    for mask_file in mask_files:
        match = pattern.search(mask_file)
        if match:
            x, y = map(int, match.groups())
        else:
            x, y = 0, 0
        mask_path = pred_mask_dir / mask_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            image_data.append((x, y, mask))

    if not image_data:
        print("No se encontraron máscaras para combinar.")
        exit()

    max_x = max(x + mask.shape[1] for x, y, mask in image_data)
    max_y = max(y + mask.shape[0] for x, y, mask in image_data)

    new_width = max_x // int(args.scale_factor)
    new_height = max_y // int(args.scale_factor)
    combined_mask = np.zeros((new_height, new_width), dtype=np.float32)

    for x, y, mask in image_data:
        small_mask = cv2.resize(mask, (mask.shape[1] // int(args.scale_factor), mask.shape[0] // int(args.scale_factor)), interpolation=cv2.INTER_NEAREST)
        x_small, y_small = x // int(args.scale_factor), y // int(args.scale_factor)
        combined_mask[y_small:y_small + small_mask.shape[0], x_small:x_small + small_mask.shape[1]] += small_mask.astype(np.float32)

    if combined_mask.max() > 0:
        combined_mask = (combined_mask / combined_mask.max()) * 255
    combined_mask = combined_mask.astype(np.uint8)

    combined_mask_path = combined_dir / "combined_mask.png"
    cv2.imwrite(str(combined_mask_path), combined_mask)
    print(f"Máscara combinada guardada en: {combined_mask_path}")

    # =====================
    # RECORTAR LA ZONA CON CONTENIDO
    # =====================
    margin = 20
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

    cropped_output_path = combined_dir / "cropped_combined_mask.png"
    cv2.imwrite(str(cropped_output_path), cropped_mask)
    print(f"Imagen combinada recortada guardada en: {cropped_output_path}")

    # =====================
    # CONTAR LOS GLOMERULOS EN LA IMAGEN COMBINADA
    # =====================
    ret, binary_cropped = cv2.threshold(cropped_mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cropped, connectivity=8)
    glomeruli_count = num_labels - 1  # Se descarta el fondo
    print(f"Número total de glomérulos en la máscara combinada: {glomeruli_count}")

    # Guardar los resultados en un archivo de texto
    result_txt_path = combined_dir / "glomeruli_count.txt"
    with open(result_txt_path, "w") as f:
        f.write("Conteo de glomérulos en cada imagen:\n")
        for img_file, count in glomeruli_counts.items():
            f.write(f"{img_file}: {count} glomérulos\n")
        f.write(f"\nNúmero total de glomérulos en la imagen combinada: {glomeruli_count}\n")

    print(f"Archivo de resultados guardado en: {result_txt_path}")

if __name__ == "__main__":
    main()
