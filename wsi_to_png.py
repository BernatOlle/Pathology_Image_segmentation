import os
import glob
import argparse
import numpy as np
import openslide
from PIL import Image
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Reducir la resolución de todas las imágenes MRXS en una carpeta")
    parser.add_argument('--input_dir', type=str, required=True, help='Carpeta con archivos MRXS')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio de salida para imágenes redimensionadas')
    return parser.parse_args()

def resize_mrxs(input_file, output_dir, max_size=2048):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(input_file)
        slide_name = Path(input_file).stem  # Nombre del archivo sin extensión
        print(slide.level_dimensions)  # Esto imprimirá los niveles disponibles
        # Obtener dimensiones en el nivel más alto de resolución
        level = 3
        width, height = slide.level_dimensions[level]
        print(f"📌 Procesando {slide_name}: {width}x{height}")

        # Calcular la nueva resolución manteniendo la relación de aspecto
        if width > height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)
            

        print(f"🔽 Redimensionando a: {new_width}x{new_height}")

        # Leer la imagen completa a la máxima resolución
        img = slide.read_region((0, 0), level, (width, height)).convert("RGB")

        # Redimensionar la imagen
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Crear carpeta de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{slide_name}_resized.png")

        # Guardar imagen redimensionada
        img_resized.save(output_path, "PNG")
        output_dir = os.path.join(output_dir, slide_name)
        output_path = os.path.join(output_dir, f"{slide_name}_resized.png")
        print(f"✅ Guardado en: {output_path}")

    except Exception as e:
        print(f"❌ Error al procesar {input_file}: {e}")

    finally:
        if 'slide' in locals():
            slide.close()

def process_folder(input_dir, output_dir):
    print(f"🔍 Buscando archivos MRXS en {input_dir}")
    mrxs_files = glob.glob(os.path.join(input_dir, "*.mrxs"))

    if not mrxs_files:
        print(f"❌ No se encontraron archivos MRXS en {input_dir}")
        return

    print(f"🔍 Se encontraron {len(mrxs_files)} archivos MRXS en {input_dir}")

    for file in mrxs_files:
      print(f"🔍 Procesando {file}")
      resize_mrxs(file, output_dir)

def main():
    print("🚀 Iniciando proceso de reducción de resolución de imágenes MRXS")
    args = parse_args()
    print(f"📁 Carpeta de entrada: {args.input_dir}")
    process_folder(args.input_dir, args.output_dir)

if __name__ == "__main__":
    print("🚀 Iniciando proceso de reducción de resolución de imágenes MRXS")
    main()