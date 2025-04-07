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
    parser.add_argument('--training_patches_dir', type=str, 
                        default="/mnt/work/datasets/BKidney/KPIS/KPIs24 Training Data/Task1_patch_level/train",
                        help='Directorio con los parches de entrenamiento')
    parser.add_argument('--compute_reference', action='store_true', 
                        help='Computar las estadísticas de referencia desde los parches de entrenamiento')
    parser.add_argument('--stats_json', type=str, 
                        default="global_stats.json", 
                        help='Ruta al JSON con estadísticas de referencia')
    parser.add_argument('--extreme_threshold', type=float, default=0.95, 
                        help='Umbral para filtrar imágenes con píxeles extremos (0-1)')
    parser.add_argument('--min_variance', type=float, default=10.0, 
                        help='Varianza mínima para considerar la imagen válida')
    parser.add_argument('--histogram_match_method', type=str, default='distribution',
                        choices=['distribution', 'moments'], 
                        help='Método de coincidencia de histograma: "distribution" para coincidir con la distribución completa, "moments" para coincidencia de media y varianza')
    return parser.parse_args()

def compute_training_histograms(training_dir):
    """
    Computa los histogramas de referencia de las imágenes de entrenamiento.
    Devuelve tanto las estadísticas de momento (media, varianza) como los histogramas completos.
    """
    print(f"Calculando estadísticas de referencia desde: {training_dir}")
    
    # Acumuladores para histogramas
    combined_hist_r = np.zeros(256)
    combined_hist_g = np.zeros(256)
    combined_hist_b = np.zeros(256)
    
    # Acumuladores para estadísticas
    r_values = []
    g_values = []
    b_values = []
    
    total_images = 0
    
    # Buscar todas las carpetas en el directorio de entrenamiento
    training_folders = [f for f in Path(training_dir).iterdir() if f.is_dir()]
    
    for folder in tqdm(training_folders, desc="Procesando carpetas de entrenamiento"):
        # Buscar subcarpetas que contengan imágenes
        for subdir in folder.glob("**/"):
            print(subdir)
            img_dir = subdir / "img"
            if img_dir.exists() and img_dir.is_dir():
                # Procesar todas las imágenes en img_dir
                for img_path in img_dir.glob("*.jpg"):
                    try:
                        # Cargar imagen
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        # Verificar que sea RGB
                        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                            continue
                        
                        # Acumular valores para estadísticas
                        r_values.extend(img_array[:,:,0].flatten())
                        g_values.extend(img_array[:,:,1].flatten())
                        b_values.extend(img_array[:,:,2].flatten())
                        
                        # Acumular histogramas
                        combined_hist_r += np.histogram(img_array[:,:,0], bins=256, range=(0,256))[0]
                        combined_hist_g += np.histogram(img_array[:,:,1], bins=256, range=(0,256))[0]
                        combined_hist_b += np.histogram(img_array[:,:,2], bins=256, range=(0,256))[0]
                        
                        total_images += 1
                    except Exception as e:
                        print(f"Error al procesar {img_path}: {e}")
    
    if total_images == 0:
        raise ValueError("No se encontraron imágenes válidas en el directorio de entrenamiento")
    
    # Convertir listas a arrays numpy para cálculos eficientes
    r_values = np.array(r_values)
    g_values = np.array(g_values)
    b_values = np.array(b_values)
    
    # Normalizar histogramas (convertir a distribución de probabilidad)
    combined_hist_r = combined_hist_r / combined_hist_r.sum()
    combined_hist_g = combined_hist_g / combined_hist_g.sum()
    combined_hist_b = combined_hist_b / combined_hist_b.sum()
    
    # Calcular CDF para cada histograma
    r_cdf = np.cumsum(combined_hist_r)
    g_cdf = np.cumsum(combined_hist_g)
    b_cdf = np.cumsum(combined_hist_b)
    
    # Calcular estadísticas
    stats = {
        "Red": {
            "mean": float(np.mean(r_values)),
            "variance": float(np.var(r_values)),
            "min": float(np.min(r_values)),
            "max": float(np.max(r_values)),
            "histogram": combined_hist_r.tolist(),
            "cdf": r_cdf.tolist()
        },
        "Green": {
            "mean": float(np.mean(g_values)),
            "variance": float(np.var(g_values)),
            "min": float(np.min(g_values)),
            "max": float(np.max(g_values)),
            "histogram": combined_hist_g.tolist(),
            "cdf": g_cdf.tolist()
        },
        "Blue": {
            "mean": float(np.mean(b_values)),
            "variance": float(np.var(b_values)),
            "min": float(np.min(b_values)),
            "max": float(np.max(b_values)),
            "histogram": combined_hist_b.tolist(),
            "cdf": b_cdf.tolist()
        },
        "metadata": {
            "total_images": total_images,
            "total_pixels": len(r_values)
        }
    }
    
    print(f"Estadísticas calculadas a partir de {total_images} imágenes")
    return stats

def match_histograms_distribution(source_img, reference_stats):
    """
    Coincidencia de histograma usando el método de transformación de ecualización.
    Utiliza las CDFs de referencia para transformar la imagen fuente.
    """
    matched_img = np.zeros_like(source_img)
    
    for channel in range(3):
        # Datos del canal fuente
        source_data = source_img[:, :, channel].astype(np.float32)
        
        # Calcular histograma y CDF de la imagen fuente
        source_hist, bin_edges = np.histogram(source_data, bins=256, range=(0, 256))
        source_hist = source_hist / source_hist.sum()
        source_cdf = np.cumsum(source_hist)
        
        # Obtener CDF de referencia
        channel_name = ['Red', 'Green', 'Blue'][channel]
        ref_cdf = np.array(reference_stats[channel_name]['cdf'])
        
        # Crear la tabla de búsqueda para la transformación
        lookup_table = np.zeros(256)
        for i in range(256):
            # Encontrar el valor en la CDF de referencia que mejor coincide con la CDF de origen
            lookup_table[i] = np.argmin(np.abs(ref_cdf - source_cdf[i]))
        
        # Aplicar la transformación a la imagen
        matched_channel = lookup_table[source_data.astype(np.uint8)]
        matched_img[:, :, channel] = matched_channel.astype(np.uint8)
    
    return matched_img

def match_histograms_moments(source_img, reference_stats, alpha=1.0):
    """
    Adapta el histograma de la imagen fuente para que coincida con las estadísticas de referencia,
    usando momentos estadísticos (media y varianza).
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
                             extreme_threshold=0.95, min_variance=10, match_method='distribution'):
    try:
        # Abrir la imagen con OpenSlide
        slide = openslide.OpenSlide(slide_path)
        slide_name = Path(slide_path).stem

        # Directorios de salida
        slide_output_dir = Path.cwd()/"result"/slide_name/"patches"
        os.makedirs(f"{slide_output_dir}/original", exist_ok=True)
        os.makedirs(f"{slide_output_dir}/matched", exist_ok=True)

        # Obtener dimensiones
        width, height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level] if level > 0 else 1

        print(f"\nProcesando {slide_name} ({width}x{height} en nivel {level})")
        print(f"Filtros: extremos > {extreme_threshold*100}% | varianza < {min_variance}")
        print(f"Método de normalización: {match_method}")

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
                    
                    # Aplicar normalización según el método elegido
                    if match_method == 'distribution':
                        matched_array = match_histograms_distribution(patch_array, reference_stats)
                    else:  # moments
                        matched_array = match_histograms_moments(patch_array, reference_stats)
                    
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

    # Computar o cargar estadísticas de referencia
    if args.compute_reference:
        try:
            reference_stats = compute_training_histograms(args.training_patches_dir)
            # Guardar estadísticas para uso futuro
            with open(args.stats_json, 'w') as f:
                json.dump(reference_stats, f, indent=2)
            print(f"✅ Estadísticas guardadas en {args.stats_json}")
        except Exception as e:
            print(f"❌ Error al computar estadísticas: {e}")
            return
    else:
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
            min_variance=args.min_variance,
            match_method=args.histogram_match_method
        )

    print("\n✅ Procesamiento completado")

if __name__ == "__main__":
    main()