import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Recortar imágenes médicas MRXS en parches de 2048x2048 con un stride configurable')
    parser.add_argument('--input_dir', type=str, required=True, help='Archivo MRXS o directorio con imágenes MRXS')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio de salida para los recortes')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024, help='Stride (paso) para el recorte de los parches (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar la imagen (default: 0, máxima resolución)')
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

def is_black_or_white(patch_array, threshold=0.90):
    """
    Determina si un parche es mayoritariamente negro o blanco.
    
    - threshold: porcentaje mínimo de píxeles (0.0 - 1.0) que deben ser negros o blancos para descartar el parche.
    - Si más del 'threshold' de los píxeles son negros (cerca de 0) o blancos (cerca de 255), se considera no válido.
    """
    # Convertir a escala de grises para simplificar el análisis
    gray_patch = np.mean(patch_array, axis=2)  # Promedio en el canal de color (RGB → escala de grises)

    # Contar cuántos píxeles son "muy negros" o "muy blancos"
    black_pixels = np.sum(gray_patch < 30)  # Píxeles casi negros
    white_pixels = np.sum(gray_patch > 225)  # Píxeles casi blancos
    total_pixels = gray_patch.size  # Total de píxeles en la imagen

    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels

    if black_ratio > threshold:
        return True
    if white_ratio > threshold:
        
        return True

    return False  # El parche es válido

def extract_patches_from_slide(slide_path, output_dir, patch_size=2048, level=0):
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

        # Calcular número de recortes en cada dimensión
        num_patches_x = width // patch_size
        num_patches_y = height // patch_size

        # Ajustar para cubrir toda la imagen
        if width % patch_size != 0:
            num_patches_x += 1
        if height % patch_size != 0:
            num_patches_y += 1

        total_patches = num_patches_x * num_patches_y
        print(f"Generando {total_patches} recortes...")

        # Extraer y guardar cada recorte
        patch_id = 0
        for y in tqdm(range(0, height, patch_size), desc="Filas"):
            for x in range(0, width, patch_size):
                # Ajustar las coordenadas si estamos en el borde
                actual_width = min(patch_size, width - x)
                actual_height = min(patch_size, height - y)

                # Solo procesar recortes completos
                if actual_width == patch_size and actual_height == patch_size:
                    # Convertir coordenadas al nivel 0 (multiplicar por factor de escala)
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)

                    # Leer el recorte de la imagen
                    patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                    patch = patch.convert("RGB")

                    # Verificar si el parche no es completamente negro o blanco
                    patch_array = np.array(patch)
                    if patch_array.std() > 10:  # Si hay variación en los valores de píxel
                        # Generar nombre del archivo
                        patch_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                        patch_path = slide_output_dir / patch_filename

                        # Guardar el recorte
                        patch.save(str(patch_path))
                        patch_id += 1

        print(f"Procesamiento completo. Se generaron {patch_id} recortes.")

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
    finally:
        if 'slide' in locals():
            slide.close()

def main():
    args = parse_args()
    
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
            level=args.level
        )
    
    print("✅ Procesamiento completo de todas las imágenes.")

if __name__ == "__main__":
    main()
