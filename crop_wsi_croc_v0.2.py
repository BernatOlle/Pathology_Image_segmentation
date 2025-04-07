import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Recortar imágenes médicas MRXS en parches de 2048x2048 con un stride configurable')
    parser.add_argument('--input_dir', type=str, required=True, help='Archivo MRXS o directorio con imágenes MRXS')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio de salida para los recortes')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024, help='Stride (paso) para el recorte de los parches (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar la imagen (default: 0, máxima resolución)')
    parser.add_argument('--tissue_threshold', type=float, default=0.05, help='Umbral mínimo de contenido de tejido (default: 0.05)')
    parser.add_argument('--blank_threshold', type=float, default=0.90, help='Umbral para descartar parches en blanco o negro (default: 0.90)')
    parser.add_argument('--normalize', action='store_true', help='Aplicar normalización de color')
    parser.add_argument('--no_color_correction', action='store_true', help='No aplicar corrección de color')
    return parser.parse_args()

def clear_output_directory(output_dir):
    """
    Elimina todo el contenido del directorio de salida antes de generar nuevos archivos.
    """
    output_path = Path(output_dir)
    if output_path.exists() and output_path.is_dir():
        for file in output_path.glob("*"):
            try:
                if file.is_file():
                    file.unlink()  # Eliminar archivos
                elif file.is_dir():
                    import shutil
                    shutil.rmtree(file)  # Eliminar carpetas y su contenido
            except Exception as e:
                print(f"❌ Error eliminando {file}: {e}")

def is_valid_patch(patch_array, blank_threshold=0.90, tissue_threshold=0.05):
    """
    Determina si un parche contiene suficiente tejido para ser considerado válido.
    
    - blank_threshold: porcentaje máximo de píxeles (0.0 - 1.0) que pueden ser negros o blancos.
    - tissue_threshold: porcentaje mínimo de tejido que debe contener el parche.
    """
    # Convertir a escala de grises para simplificar el análisis
    gray_patch = cv2.cvtColor(patch_array, cv2.COLOR_RGB2GRAY)
    
    # Verificar si el parche es mayoritariamente negro o blanco
    black_pixels = np.sum(gray_patch < 30)  # Píxeles casi negros
    white_pixels = np.sum(gray_patch > 225)  # Píxeles casi blancos
    total_pixels = gray_patch.size  # Total de píxeles en la imagen

    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    
    # Si supera el umbral de píxeles blancos o negros, descartar
    if black_ratio > blank_threshold or white_ratio > blank_threshold:
        return False
        
    # Detectar tejido usando umbralización
    # Para tejido H&E, aplicamos umbral para capturar áreas teñidas
    _, tissue_mask = cv2.threshold(gray_patch, 210, 255, cv2.THRESH_BINARY_INV)
    tissue_pixels = np.sum(tissue_mask > 0)
    tissue_ratio = tissue_pixels / total_pixels
    
    # Verificar si hay suficiente tejido
    return tissue_ratio >= tissue_threshold

def normalize_staining(img):
    """
    Normaliza la tinción H&E basado en el método de Macenko.
    Simplificado para este script.
    """
    # Separar canales
    img = img.astype(np.float32)
    img = np.maximum(img, 1.0)  # Evitar log(0)
    
    # Convertir a valores ópticos (densidad óptica)
    od = -np.log(img / 255.0)
    
    # Eliminar pixeles con poca densidad
    od_flat = od.reshape((-1, 3))
    od_threshold = 0.15
    tissue_mask = np.sum(od_flat, axis=1) > od_threshold
    
    if tissue_mask.sum() < 100:  # Si no hay suficiente tejido, devolver la imagen original
        return img.astype(np.uint8)
        
    # Valores objetivo para H&E normalizado
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    
    # Aplicar normalización simple
    od_norm = np.zeros_like(od)
    for i in range(3):  # Para cada canal (R,G,B)
        od_norm[:,:,i] = (od[:,:,i] - np.min(od[:,:,i])) / (np.max(od[:,:,i]) - np.min(od[:,:,i]) + 1e-8) * 255
        
    # Convertir de vuelta a RGB
    img_norm = np.exp(-od_norm / 255 * np.log(255))
    
    # Ajustar intensidad para resaltar las estructuras H&E
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    
    return img_norm

def color_correction(patch_array):
    """
    Aplica corrección de color para resaltar las características de la tinción H&E.
    """
    # Aumentar contraste
    lab = cv2.cvtColor(patch_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Ecualizar el canal L (luminosidad)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    
    # Mejorar los canales a y b para resaltar la tinción H&E
    a = cv2.add(a, 5)  # Aumentar ligeramente el componente verde-rojo
    b = cv2.add(b, 5)  # Aumentar ligeramente el componente azul-amarillo
    
    # Combinar canales y convertir de vuelta a RGB
    lab_enhanced = cv2.merge((l_eq, a, b))
    enhanced_img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_img

def extract_patches_from_slide(slide_path, output_dir, patch_size=2048, stride=1024, 
                              level=0, blank_threshold=0.90, tissue_threshold=0.05,
                              normalize=False, no_color_correction=False):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)

        # Obtener el nombre base de la imagen sin extensión
        slide_name = Path(slide_path).stem

        # Crear carpeta específica para esta imagen
        slide_output_dir = Path(output_dir) / slide_name
        os.makedirs(slide_output_dir, exist_ok=True)

        # Obtener dimensiones de la imagen en el nivel especificado
        width, height = slide.level_dimensions[level]

        print(f"Procesando {slide_name}")
        print(f"Dimensiones: {width}x{height} en nivel {level}")

        # Calcular el factor de escala entre el nivel 0 y el nivel seleccionado
        if level > 0:
            downsample = slide.level_downsamples[level]
        else:
            downsample = 1

        # Calcular el número de parches según el stride
        num_patches_x = (width - patch_size) // stride + 2
        num_patches_y = (height - patch_size) // stride + 2
        
        total_patches = num_patches_x * num_patches_y
        valid_patches = 0
        print(f"Analizando aproximadamente {total_patches} posiciones...")

        # Extraer y guardar cada recorte
        patch_id = 0
        for y in tqdm(range(0, height - patch_size + stride, stride), desc="Filas"):
            if y + patch_size > height:
                y = height - patch_size  # Ajustar para no exceder los límites
                
            for x in range(0, width - patch_size + stride, stride):
                if x + patch_size > width:
                    x = width - patch_size  # Ajustar para no exceder los límites
                
                # Convertir coordenadas al nivel 0 si es necesario
                x0 = int(x * downsample)
                y0 = int(y * downsample)

                # Leer el recorte de la imagen
                patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                patch = patch.convert("RGB")

                # Convertir a array numpy
                patch_array = np.array(patch)
                
                # Verificar si el parche es válido (contiene suficiente tejido)
                if is_valid_patch(patch_array, blank_threshold, tissue_threshold):
                    # Aplicar procesamiento de color según las opciones
                    if normalize:
                        # Normalización de tinción
                        patch_array = normalize_staining(patch_array)
                    elif not no_color_correction:
                        # Corrección de color para resaltar las características H&E
                        patch_array = color_correction(patch_array)
                    
                    # Convertir de nuevo a imagen PIL
                    patch = Image.fromarray(patch_array.astype(np.uint8))

                    # Generar nombre del archivo
                    patch_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                    patch_path = slide_output_dir / patch_filename

                    # Guardar el recorte
                    patch.save(str(patch_path))
                    valid_patches += 1
                    patch_id += 1

        print(f"Procesamiento completo. Se generaron {valid_patches} parches válidos de {total_patches} posiciones analizadas.")

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
    finally:
        if 'slide' in locals():
            slide.close()

def main():
    args = parse_args()
    
    # Verificar que OpenCV esté disponible
    try:
        import cv2
    except ImportError:
        print("❌ Error: OpenCV (cv2) no está instalado. Instálelo con 'pip install opencv-python'")
        return
    
    # Eliminar contenido anterior de la carpeta de salida
    clear_output_directory(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

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

    print(f"🔍 Se encontraron {len(mrxs_files)} archivos MRXS")

    for slide_path in mrxs_files:
        extract_patches_from_slide(
            slide_path, 
            args.output_dir, 
            patch_size=args.patch_size,
            stride=args.stride,
            level=args.level,
            blank_threshold=args.blank_threshold,
            tissue_threshold=args.tissue_threshold,
            normalize=args.normalize,
            no_color_correction=args.no_color_correction
        )
    
    print("✅ Procesamiento completo de todas las imágenes.")

if __name__ == "__main__":
    main()