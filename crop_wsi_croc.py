import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import openslide
from PIL import Image
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Recortar imágenes médicas MRXS en parches con normalización de histograma'
    )
    parser.add_argument('--input_dir', type=str, required=True, help='Archivo MRXS o directorio con imágenes MRXS')
    parser.add_argument('--patch_size', type=int, default=2048, help='Tamaño de los recortes (default: 2048)')
    parser.add_argument('--stride', type=int, default=1024, help='Stride (paso) para el recorte (default: 1024)')
    parser.add_argument('--level', type=int, default=0, help='Nivel de zoom para procesar (default: 0)')
    parser.add_argument('--stats_json', type=str,default="/home/usuaris/imatge/bernat.olle/wsi_glomerulus_seg/scripts_CB/test/histogram/global_stats.json", help='Ruta al JSON con estadísticas de referencia')
    parser.add_argument('--extreme_threshold', type=float, default=0.95, help='Umbral para filtrar imágenes con píxeles extremos (0-1)')
    parser.add_argument('--min_variance', type=float, default=10.0, help='Varianza mínima para considerar la imagen válida')
    return parser.parse_args()

def match_histograms(source_img, reference_stats, alpha=1.0):
    """
    Adapta el histograma de la imagen fuente para que coincida con las estadísticas de referencia,
    pero con una transformación no lineal que preserva los extremos (blancos y negros).
    
    Parámetros:
        source_img: Imagen fuente (en formato uint8, H x W x 3).
        reference_stats: Diccionario con medias y varianzas de referencia.
        alpha: Controla la "fuerza" de la no linealidad (alpha > 0). 
               Si alpha=1, se comporta como la versión lineal original.
    """
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):  # Para cada canal RGB
        # Datos del canal fuente (convertir a float32)
        source_data = source_img[:, :, channel].astype(np.float32)
        
        # Normalizar a [0, 1] (opcional, para trabajar en rango manejable)
        source_normalized = source_data / 255.0
        
        # Aplicar función sigmoidal (no linealidad controlada por alpha)
        # Usamos una tangente hiperbólica modificada para empujar valores hacia los extremos
        source_nonlinear = np.tanh(alpha * (2 * source_normalized - 1))  # Mapea [0,1] -> [-1,1]
        source_nonlinear = (source_nonlinear + 1) / 2  # Volver a [0,1]
        
        # Reescalar a [0, 255] antes del ajuste lineal (opcional)
        source_nonlinear = source_nonlinear * 255.0
        
        # Calcular estadísticas de la imagen no lineal
        source_mean = np.mean(source_nonlinear)
        source_std = np.std(source_nonlinear)
        
        # Estadísticas de referencia
        channel_name = ['Red', 'Green', 'Blue'][channel]
        ref_mean = reference_stats[channel_name]['mean']
        ref_std = np.sqrt(reference_stats[channel_name]['variance'])
        
        # Evitar división por cero
        if source_std == 0:
            source_std = 1
        
        # Aplicar transformación lineal (ajuste de media y std)
        matched_data = (source_nonlinear - source_mean) * (ref_std / source_std) + ref_mean
        
        # Asegurar valores dentro del rango [0, 255]
        matched_data = np.clip(matched_data, 0, 255)
        matched_img[:, :, channel] = matched_data.astype(np.uint8)
    
    return matched_img

def is_extreme_image(patch_array, extreme_threshold=0.95, min_variance=10):
    """
    Determina si una imagen tiene demasiados píxeles en los extremos (0 o 255)
    o muy poca varianza
    """
    # Calcular varianza global
    if patch_array.std() < min_variance:
        return True
    
    # Analizar histograma para cada canal
    for channel in range(3):
        hist = np.histogram(patch_array[:,:,channel].ravel(), bins=256, range=(0,256))[0]
        total_pixels = hist.sum()
        extreme_pixels = hist[0] + hist[-1]  # Píxeles en 0 y 255
        
        if extreme_pixels / total_pixels > extreme_threshold:
            return True
    
    return False

def extract_patches_from_slide(slide_path, patch_size=2048, level=0, reference_stats=None, 
                             extreme_threshold=0.95, min_variance=10):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem

        # Directorios de salida
        slide_output_dir = Path.cwd()/"result"/slide_name/"paches"
        os.makedirs(f"{slide_output_dir}/original", exist_ok=True)
        os.makedirs(f"{slide_output_dir}/matched", exist_ok=True)

        # Obtener dimensiones
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level] if level > 0 else 1

        print(f"\nProcesando {slide_name} ({width}x{height} en nivel {level})")
        print(f"Filtros: extremos > {extreme_threshold*100}% | varianza < {min_variance}")

        patch_id = 0
        valid_patches = 0
        rejected_patches = 0
        
        for y in tqdm(range(0, height, patch_size), desc="Procesando filas"):
            for x in range(0, width, patch_size):
                actual_width = min(patch_size, width - x)
                actual_height = min(patch_size, height - y)

                # Solo recortes completos
                if actual_width == patch_size and actual_height == patch_size:
                    x0 = int(x * downsample)
                    y0 = int(y * downsample)
                    patch = slide.read_region((x0, y0), level, (patch_size, patch_size))
                    patch = patch.convert("RGB")
                    patch_array = np.array(patch)
                    
                    # Verificar si el parche es válido
                    if is_extreme_image(patch_array, extreme_threshold, min_variance):
                        rejected_patches += 1
                        continue
                    
                    # Guardar original
                    patch_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}.png"
                    Image.fromarray(patch_array).save(slide_output_dir/"original"/patch_filename)
                    
                    # Aplicar normalización
                    matched_array = match_histograms(patch_array, reference_stats)
                    
                    # Guardar normalizado
                    matched_filename = f"{slide_name}_patch{patch_id:04d}_x{x}_y{y}_matched.png"
                    Image.fromarray(matched_array).save(slide_output_dir/"matched"/matched_filename)
                    
                    patch_id += 1
                    valid_patches += 1

        print(f"Procesamiento completo. Parches válidos: {valid_patches} | Rechazados: {rejected_patches}")

    except Exception as e:
        print(f"Error al procesar {slide_path}: {e}")
    finally:
        if 'slide' in locals():
            slide.close()

def main():
    args = parse_args()
    input_path = Path(args.input_dir)

    # Cargar estadísticas de referencia
    try:
        with open(args.stats_json, 'r') as f:
            reference_stats = json.load(f)
        print(f"✅ Estadísticas cargadas de {args.stats_json}")
    except Exception as e:
        print(f"❌ Error al cargar el JSON: {e}")
        return

    # Buscar archivos MRXS
    if input_path.is_file() and input_path.suffix.lower() == ".mrxs":
        mrxs_files = [str(input_path)]
    elif input_path.is_dir():
        mrxs_files = glob.glob(os.path.join(args.input_dir, "*.mrxs"))
    else:
        print(f"❌ Ruta inválida: {args.input_dir}")
        return

    if not mrxs_files:
        print(f"❌ No se encontraron archivos MRXS en {args.input_dir}")
        return

    print(f"🔍 Encontrados {len(mrxs_files)} archivos MRXS")

    # Procesar cada archivo
    for slide_path in mrxs_files:
        extract_patches_from_slide(
            slide_path,
            patch_size=args.patch_size,
            level=args.level,
            reference_stats=reference_stats,
            extreme_threshold=args.extreme_threshold,
            min_variance=args.min_variance
        )

    print("\n✅ Procesamiento completado")

if __name__ == "__main__":
    main()