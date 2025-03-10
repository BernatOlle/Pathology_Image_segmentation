import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide

def parse_args():
    parser = argparse.ArgumentParser(description='Recortar imágenes médicas MRXS en parches de 2048x2048')
    parser.add_argument('--input_dir', type=str, required=True, help='Directorio con las imágenes MRXS')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio de salida para los recortes')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar la imagen (default: 0, máxima resolución)')
    return parser.parse_args()

def extract_patches_from_slide(slide_path, output_dir, patch_size=2048, level=0):
    """
    Extrae recortes de tamaño patch_size x patch_size de una imagen MRXS
    
    Args:
        slide_path: Ruta a la imagen MRXS
        output_dir: Directorio donde guardar los recortes
        patch_size: Tamaño de los recortes (default: 2048)
        level: Nivel de zoom (default: 0, máxima resolución)
    """
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
                    # Leer el recorte de la imagen
                    patch = slide.read_region((x, y), level, (patch_size, patch_size))
                    patch = patch.convert("RGB")
                    
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
    
    # Crear directorio de salida principal
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Buscar todas las imágenes MRXS en el directorio de entrada
    mrxs_files = glob.glob(os.path.join(args.input_dir, "*.mrxs"))
    
    if not mrxs_files:
        print(f"No se encontraron archivos MRXS en {args.input_dir}")
        return
    
    print(f"Se encontraron {len(mrxs_files)} archivos MRXS")
    
    # Procesar cada imagen
    for slide_path in mrxs_files:
        extract_patches_from_slide(
            slide_path, 
            args.output_dir, 
            patch_size=args.patch_size, 
            level=args.level
        )
    
    print("Procesamiento completo de todas las imágenes.")

if __name__ == "__main__":
    main()