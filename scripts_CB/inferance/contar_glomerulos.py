import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage, stats
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import json
import uuid
from skimage.measure import find_contours

def setup_logger(name="glomeruli_calibrator"):
    """Configura el logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class GlomeruliParameterCalibrator:
    """
    Clase para calibrar parámetros de detección de glomérulos.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Inicializa el calibrador.
        
        Args:
            output_dir: Directorio para guardar resultados
        """
        self.logger = setup_logger()
        self.image_path = None
        self.output_dir = None
        
        self.image = None
        self.analysis_results = {}
        
        # Rangos de parámetros para probar
        self.kernel_sizes = [3, 5, 7, 9]
        self.min_distances = [10, 15, 20, 25, 30, 40, 50]
        self.min_areas = [25, 50, 100, 200, 500, 1000]
    
    def process_complete(self, image_path: str, output_dir: str = None, downsampling: int = 2, result: dict = None) -> Dict:
        """
        Ejecuta el proceso completo de calibración.
        
        Args:
            image_path: Ruta de la imagen a procesar
            output_dir: Directorio para guardar resultados (opcional)
            downsampling: Factor de reducción de la imagen (por defecto 2)
            result: Resultados previos (opcional)
            
        Returns:
            dict: Resultados completos del análisis
        """
        # Configurar rutas
        self.image_path = Path(image_path)
        self.output_dir = Path(output_dir) if output_dir else self.image_path.parent / "calibration"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result = result
        
        try:
            self.logger.info(f"Iniciando calibración para: {self.image_path}")
            
            # Cargar y preparar imagen con downsampling
            if not self.load_and_prepare_image(downsampling):
                return {'error': 'No se pudo cargar la imagen'}
            
            self.logger.info("✅ Imagen cargada correctamente")
            
            # Ejecutar análisis
            self.logger.info("🔍 Analizando operaciones morfológicas...")
            self.analyze_morphological_operations()
            
            self.logger.info("📊 Analizando distribución de áreas...")
            self.analyze_area_distribution(kernel_size=5)
            
            self.logger.info("🧪 Probando parámetros de watershed...")
            self.test_watershed_parameters(kernel_size=5)
            
            self.logger.info("📋 Generando reporte...")
            report = self.generate_calibration_report()
            
            # Visualizar con parámetros óptimos
            if 'watershed_parameters' in self.analysis_results:
                optimal_params = self.analysis_results['watershed_parameters']['optimal_params']
                self.logger.info("🖼️ Generando visualización con parámetros óptimos...")
                
                count, regions, geojson_path = self.visualize_detection_with_params(
                    min_distance=optimal_params['best_balance']['min_distance'],
                    min_area=optimal_params['best_balance']['min_area']
                )
                
                # Agregar información de visualización al reporte
                report['visualization'] = {
                    'optimal_params': optimal_params['best_balance'],
                    'detected_count': count,
                    'geojson_path': str(geojson_path)
                }
                
                self.logger.info(f"🎯 PARÁMETROS ÓPTIMOS:")
                self.logger.info(f"   Min Distance: {optimal_params['best_balance']['min_distance']}")
                self.logger.info(f"   Min Area: {optimal_params['best_balance']['min_area']}")
                self.logger.info(f"   Glomérulos esperados: {optimal_params['best_balance']['expected_count']}")
            
            self.logger.info(f"✅ Análisis completado. Resultados guardados en: {self.output_dir}")
            
            return {
                'success': True,
                'results': report,
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error durante el procesamiento: {str(e)}")
            return {'error': str(e)}
        
    def load_and_prepare_image(self, downsampling: int = 2) -> bool:
        """
        Carga y prepara la imagen para análisis.
        
        Args:
            downsampling: Factor de reducción de la imagen (por defecto 2)
            
        Returns:
            bool: True si se cargó correctamente
        """
        try:
            # Cargar imagen 
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            
            img = Image.open(self.image_path)
            self.original_size = img.size  # Guardar tamaño original (width, height)
            self.logger.info(f"Imagen original: {self.original_size}")
            
            # Aplicar downsampling
            if downsampling > 1:
                new_size = (img.width // downsampling, img.height // downsampling)
                self.scale_factor = 1.0 / downsampling  # Factor de escala inverso
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.logger.info(f"Imagen reducida con factor {downsampling} a: {new_size}")
            else:
                self.scale_factor = 1.0  # Sin reducción
                self.logger.info("Sin reducción aplicada")
                
            if img.mode != 'L':
                img = img.convert('L')
            self.image = np.array(img)
            
            # Extraer ROI (comentado por ahora para evitar problemas)
            # Si necesitas usar ROI, descomenta la siguiente línea:
            # self.image = self._extract_roi(self.image)
            
            self.logger.info(f"Imagen procesada: {self.image.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando imagen: {e}")
            return False
        
    def _extract_roi(self, image: np.ndarray, margin: int = 100) -> np.ndarray:
        """Extrae la región de interés eliminando el fondo negro y guarda el offset."""
        non_zero = np.where(image > 0)
        
        if len(non_zero[0]) == 0:
            self.roi_offset = (0, 0)  # No hay offset si no se recorta
            return image
        
        y_min, y_max = non_zero[0].min(), non_zero[0].max()
        x_min, x_max = non_zero[1].min(), non_zero[1].max()
        
        y_min = max(0, y_min - margin)
        y_max = min(image.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(image.shape[1], x_max + margin)
        
        # Guardar el offset para usar en la conversión de coordenadas
        self.roi_offset = (x_min, y_min)
        
        return image[y_min:y_max, x_min:x_max]
    
    def analyze_morphological_operations(self) -> Dict:
        """
        Analiza el efecto de diferentes kernels morfológicos.
        
        Returns:
            dict: Resultados del análisis morfológico
        """
        self.logger.info("Analizando operaciones morfológicas...")
        
        results = {}
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        
        fig, axes = plt.subplots(2, len(self.kernel_sizes), figsize=(20, 10))
        
        for i, kernel_size in enumerate(self.kernel_sizes):
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Operación de apertura
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Operación de cierre
            #closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Contar componentes conectados
            num_labels_open, labels_open = cv2.connectedComponents(opened)
            #num_labels_close, labels_close = cv2.connectedComponents(closed)
            
            results[kernel_size] = {
                'components_after_opening': num_labels_open - 1,
                #'components_after_closing': num_labels_close - 1,
                'total_area_opening': np.sum(opened > 0)
                #'total_area_closing': np.sum(closed > 0)
            }
            
            # Visualizar
            axes[0, i].imshow(opened, cmap='gray')
            axes[0, i].set_title(f'Apertura K={kernel_size}\nComp: {num_labels_open-1}')
            axes[0, i].axis('off')
            
            #axes[1, i].imshow(closed, cmap='gray')
            #axes[1, i].set_title(f'Cierre K={kernel_size}\nComp: {num_labels_close-1}')
            #axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'morphological_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['morphological'] = results
        return results
    
    def analyze_area_distribution(self, kernel_size: int = 5) -> Dict:
        """
        Analiza la distribución de áreas de los componentes detectados.
        
        Args:
            kernel_size: Tamaño de kernel para operaciones morfológicas
            
        Returns:
            dict: Estadísticas de distribución de áreas
        """
        self.logger.info("Analizando distribución de áreas...")
        
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Obtener regiones y sus propiedades
        labeled_regions = label(opened)
        regions = regionprops(labeled_regions)
        
        areas = [region.area for region in regions]
        
        if not areas:
            self.logger.warning("No se encontraron regiones")
            return {}
        
        # Estadísticas descriptivas
        areas_array = np.array(areas)
        stats_dict = {
            'count': len(areas),
            'mean': np.mean(areas_array),
            'median': np.median(areas_array),
            'std': np.std(areas_array),
            'min': np.min(areas_array),
            'max': np.max(areas_array),
            'q25': np.percentile(areas_array, 25),
            'q75': np.percentile(areas_array, 75),
            'iqr': np.percentile(areas_array, 75) - np.percentile(areas_array, 25)
        }
        
        # Detectar outliers usando IQR
        iqr = stats_dict['iqr']
        q1, q3 = stats_dict['q25'], stats_dict['q75']
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_low = areas_array[areas_array < lower_bound]
        outliers_high = areas_array[areas_array > upper_bound]
        normal_areas = areas_array[(areas_array >= lower_bound) & (areas_array <= upper_bound)]
        
        stats_dict.update({
            'outliers_low_count': len(outliers_low),
            'outliers_high_count': len(outliers_high),
            'normal_count': len(normal_areas),
            'suggested_min_area': max(50, int(q1 - 0.5 * iqr)) if len(normal_areas) > 0 else 50,
            'suggested_max_area': int(q3 + 2 * iqr) if len(normal_areas) > 0 else 10000
        })
        
        # Visualización
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histograma de áreas
        axes[0, 0].hist(areas, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats_dict['mean'], color='red', linestyle='--', label=f'Media: {stats_dict["mean"]:.1f}')
        axes[0, 0].axvline(stats_dict['median'], color='green', linestyle='--', label=f'Mediana: {stats_dict["median"]:.1f}')
        axes[0, 0].set_xlabel('Área (píxeles)')
        axes[0, 0].set_ylabel('Frecuencia')
        axes[0, 0].set_title('Distribución de Áreas')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(areas, vert=True)
        axes[0, 1].set_ylabel('Área (píxeles)')
        axes[0, 1].set_title('Box Plot de Áreas')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log-scale histogram
        axes[1, 0].hist(areas, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('Área (píxeles)')
        axes[1, 0].set_ylabel('Frecuencia (log)')
        axes[1, 0].set_title('Distribución de Áreas (Escala Log)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Área vs índice de región
        region_indices = range(len(areas))
        axes[1, 1].scatter(region_indices, areas, alpha=0.6)
        axes[1, 1].axhline(stats_dict['suggested_min_area'], color='red', linestyle='--', 
                          label=f'Min sugerida: {stats_dict["suggested_min_area"]}')
        axes[1, 1].set_xlabel('Índice de Región')
        axes[1, 1].set_ylabel('Área (píxeles)')
        axes[1, 1].set_title('Áreas por Región')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'area_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar datos de áreas para análisis posterior
        areas_df = pd.DataFrame({
            'area': areas,
            'is_outlier_low': [a < lower_bound for a in areas],
            'is_outlier_high': [a > upper_bound for a in areas]
        })
        areas_df.to_csv(self.output_dir / 'areas_data.csv', index=False)
        
        self.analysis_results['area_distribution'] = {
            'stats': stats_dict,
            'areas': areas,
            'regions_count': len(regions)
        }
        
        return stats_dict
    
    def test_watershed_parameters(self, kernel_size: int = 5) -> Dict:
        """
        Prueba diferentes combinaciones de parámetros para watershed.
        
        Args:
            kernel_size: Tamaño de kernel para operaciones morfológicas
            
        Returns:
            dict: Resultados de las pruebas de parámetros
        """
        self.logger.info("Probando parámetros de watershed...")
        
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Calcular transformada de distancia una sola vez
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        results = []
        
        # Crear una submuestra de parámetros para evitar demasiadas combinaciones
        min_distances_subset = [15, 20, 30, 40]
        min_areas_subset = [50, 100, 200, 500]
        
        for min_distance in min_distances_subset:
            for min_area in min_areas_subset:
                try:
                    # Encontrar máximos locales
                    coordinates = peak_local_max(
                        dist_transform, 
                        min_distance=min_distance,
                        threshold_abs=0.3 * np.max(dist_transform)
                    )
                    
                    if len(coordinates) == 0:
                        continue
                    
                    # Crear marcadores
                    markers = np.zeros(dist_transform.shape, dtype=np.int32)
                    for i, (y, x) in enumerate(coordinates):
                        markers[y, x] = i + 1
                    
                    # Aplicar watershed
                    labels = watershed(-dist_transform, markers, mask=opened)
                    
                    # Analizar regiones
                    regions = regionprops(labels)
                    valid_regions = [r for r in regions if r.area >= min_area]
                    
                    # Calcular métricas
                    total_detected = len(regions)
                    valid_detected = len(valid_regions)
                    areas = [r.area for r in valid_regions]
                    
                    result = {
                        'min_distance': min_distance,
                        'min_area': min_area,
                        'total_detected': total_detected,
                        'valid_detected': valid_detected,
                        'mean_area': np.mean(areas) if areas else 0,
                        'std_area': np.std(areas) if areas else 0,
                        'markers_found': len(coordinates)
                    }
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Error con min_distance={min_distance}, min_area={min_area}: {e}")
                    continue
        
        # Convertir a DataFrame para análisis
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Guardar resultados
            results_df.to_csv(self.output_dir / 'watershed_parameters_test.csv', index=False)
            
            # Visualización
            self._plot_parameter_results(results_df)
            
            # Encontrar parámetros óptimos
            optimal_params = self._find_optimal_parameters(results_df)
            
            self.analysis_results['watershed_parameters'] = {
                'results_df': results_df,
                'optimal_params': optimal_params
            }
            
            return {
                'results': results,
                'optimal_params': optimal_params,
                'results_df': results_df
            }
        else:
            self.logger.error("No se pudieron obtener resultados válidos")
            return {}
    
    def _plot_parameter_results(self, results_df: pd.DataFrame):
        """Visualiza los resultados de las pruebas de parámetros."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pivot para heatmaps
        pivot_valid = results_df.pivot(index='min_distance', columns='min_area', values='valid_detected')
        pivot_total = results_df.pivot(index='min_distance', columns='min_area', values='total_detected')
        
        # Heatmap de glomérulos válidos detectados
        sns.heatmap(pivot_valid, annot=True, fmt='d', cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Glomérulos Válidos Detectados')
        axes[0, 0].set_xlabel('Área Mínima')
        axes[0, 0].set_ylabel('Distancia Mínima')
        
        # Heatmap de total detectados
        sns.heatmap(pivot_total, annot=True, fmt='d', cmap='plasma', ax=axes[0, 1])
        axes[0, 1].set_title('Total de Regiones Detectadas')
        axes[0, 1].set_xlabel('Área Mínima')
        axes[0, 1].set_ylabel('Distancia Mínima')
        
        # Gráfico de líneas: efecto de min_distance
        for min_area in results_df['min_area'].unique():
            subset = results_df[results_df['min_area'] == min_area]
            axes[1, 0].plot(subset['min_distance'], subset['valid_detected'], 
                           marker='o', label=f'Min_area={min_area}')
        axes[1, 0].set_xlabel('Distancia Mínima')
        axes[1, 0].set_ylabel('Glomérulos Válidos')
        axes[1, 0].set_title('Efecto de la Distancia Mínima')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico de líneas: efecto de min_area
        for min_distance in results_df['min_distance'].unique():
            subset = results_df[results_df['min_distance'] == min_distance]
            axes[1, 1].plot(subset['min_area'], subset['valid_detected'], 
                           marker='s', label=f'Min_dist={min_distance}')
        axes[1, 1].set_xlabel('Área Mínima')
        axes[1, 1].set_ylabel('Glomérulos Válidos')
        axes[1, 1].set_title('Efecto del Área Mínima')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _find_optimal_parameters(self, results_df: pd.DataFrame) -> Dict:
        """Encuentra los parámetros óptimos basado en múltiples criterios."""
        # Criterio 1: Máximo número de glomérulos válidos
        max_valid_idx = results_df['valid_detected'].idxmax()
        max_valid_params = results_df.loc[max_valid_idx]
        
        # Criterio 2: Mejor balance entre detección y estabilidad
        # (penalizar desviación estándar alta en áreas)
        results_df['stability_score'] = (
            results_df['valid_detected'] / 
            (1 + results_df['std_area'] / results_df['mean_area'].replace(0, 1))
        )
        best_balance_idx = results_df['stability_score'].idxmax()
        best_balance_params = results_df.loc[best_balance_idx]
        
        # Criterio 3: Parámetros medianos (conservadores)
        median_valid = results_df['valid_detected'].median()
        closest_to_median = results_df.iloc[(results_df['valid_detected'] - median_valid).abs().argsort()[:1]]
        median_params = closest_to_median.iloc[0]
        
        optimal_params = {
            'max_detection': {
                'min_distance': int(max_valid_params['min_distance']),
                'min_area': int(max_valid_params['min_area']),
                'expected_count': int(max_valid_params['valid_detected']),
                'mean_area': float(max_valid_params['mean_area'])
            },
            'best_balance': {
                'min_distance': int(best_balance_params['min_distance']),
                'min_area': int(best_balance_params['min_area']),
                'expected_count': int(best_balance_params['valid_detected']),
                'mean_area': float(best_balance_params['mean_area']),
                'stability_score': float(best_balance_params['stability_score'])
            },
            'conservative': {
                'min_distance': int(median_params['min_distance']),
                'min_area': int(median_params['min_area']),
                'expected_count': int(median_params['valid_detected']),
                'mean_area': float(median_params['mean_area'])
            }
        }
        
        return optimal_params
    
    

    def visualize_detection_with_params(self, min_distance: int, min_area: int, kernel_size: int = 5):
        """
        Visualiza la detección con parámetros específicos y genera GeoJSON.
        
        Args:
            min_distance: Distancia mínima entre marcadores
            min_area: Área mínima para glomérulos válidos
            kernel_size: Tamaño del kernel morfológico
        """
        self.logger.info(f"Visualizando detección con min_distance={min_distance}, min_area={min_area}")
        
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Transformada de distancia
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Encontrar máximos locales
        coordinates = peak_local_max(
            dist_transform, 
            min_distance=min_distance,
            threshold_abs=0.3 * np.max(dist_transform)
        )
        
        # Crear marcadores
        markers = np.zeros(dist_transform.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates):
            markers[y, x] = i + 1
        
        # Aplicar watershed
        labels = watershed(-dist_transform, markers, mask=opened)
        
        # Filtrar por área
        regions = regionprops(labels)
        valid_regions = [r for r in regions if r.area >= min_area]
        
        # Generar GeoJSON
        geojson_data = self._generate_geojson(valid_regions, labels, self.result)
        
        # Guardar GeoJSON
        geojson_filename = f'glomeruli_dist{min_distance}_area{min_area}.geojson'
        geojson_path = self.output_dir / geojson_filename
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        self.logger.info(f"GeoJSON guardado en: {geojson_path}")
        
        # Crear visualización
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Imagen original ROI
        axes[0, 0].imshow(self.image, cmap='gray')
        axes[0, 0].set_title('ROI Original')
        axes[0, 0].axis('off')
        
        # Imagen después de apertura
        axes[0, 1].imshow(opened, cmap='gray')
        axes[0, 1].set_title(f'Después de Apertura (K={kernel_size})')
        axes[0, 1].axis('off')
        
        # Transformada de distancia
        axes[0, 2].imshow(dist_transform, cmap='hot')
        axes[0, 2].set_title('Transformada de Distancia')
        axes[0, 2].axis('off')
        
        # Marcadores
        marker_display = np.zeros_like(self.image)
        for y, x in coordinates:
            marker_display[y, x] = 255
        axes[1, 0].imshow(self.image, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(marker_display, cmap='Reds', alpha=0.5)
        axes[1, 0].set_title(f'Marcadores (n={len(coordinates)})')
        axes[1, 0].axis('off')
        
        # Todas las regiones
        if np.max(labels) > 0:
            label_viz = (255 * labels / np.max(labels)).astype(np.uint8)
            label_viz_color = cv2.applyColorMap(label_viz, cv2.COLORMAP_JET)
            label_viz_color[labels == 0] = [0, 0, 0]
            axes[1, 1].imshow(cv2.cvtColor(label_viz_color, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'Todas las Regiones (n={len(regions)})')
        axes[1, 1].axis('off')
        
        # Solo regiones válidas
        valid_labels = np.zeros_like(labels)
        for i, region in enumerate(valid_regions):
            valid_labels[labels == region.label] = i + 1
        
        if np.max(valid_labels) > 0:
            valid_viz = (255 * valid_labels / np.max(valid_labels)).astype(np.uint8)
            valid_viz_color = cv2.applyColorMap(valid_viz, cv2.COLORMAP_VIRIDIS)
            valid_viz_color[valid_labels == 0] = [0, 0, 0]
            axes[1, 2].imshow(cv2.cvtColor(valid_viz_color, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Glomérulos Válidos (n={len(valid_regions)})')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        filename = f'detection_visualization_dist{min_distance}_area{min_area}.png'
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return len(valid_regions), valid_regions, geojson_path

    def _generate_geojson(self, valid_regions, labels, result=None):
        """
        Genera un archivo GeoJSON a partir de las regiones válidas, 
        escalando las coordenadas a las dimensiones originales considerando
        múltiples niveles de downsampling y ajustándolas respecto a la bounding box de recorte.
        
        Args:
            valid_regions: Regiones válidas detectadas
            labels: Array de etiquetas
            result: Diccionario con información de la bounding box, downsample_info y patch_info
        """
        features = []
        
        # Extraer información de la bounding box y downsampling si está disponible
        bbox_offset = None
        total_downsample_factor = 1
        
        if result and 'bounding_box' in result:
            bbox_info = result['bounding_box']
            bbox_offset = (bbox_info['x'], bbox_info['y'])
            
            # Obtener factor de downsampling del resultado previo
            if 'downsample_info' in result:
                prev_downsample = result['downsample_info']['factor']
                # Combinar con el factor actual de escala
                current_scale_factor = getattr(self, 'scale_factor', 1.0)
                # El factor total es la combinación de ambos downsamplings
                total_downsample_factor = prev_downsample / current_scale_factor
            else:
                total_downsample_factor = 1.0 / getattr(self, 'scale_factor', 1.0)
        else:
            total_downsample_factor = 1.0 / getattr(self, 'scale_factor', 1.0)
        
        for region in valid_regions:
            region_mask = (labels == region.label).astype(np.uint8)
            contours = find_contours(region_mask, 0.5)
            
            if len(contours) > 0:
                main_contour = max(contours, key=len)
                coordinates = []
                
                for point in main_contour:
                    y, x = point  # find_contours devuelve (y, x)
                    
                    # PRIMERO: Escalar coordenadas considerando todos los downsamplings
                    if total_downsample_factor != 1.0:
                        x *= total_downsample_factor
                        y *= total_downsample_factor
                    
                    # SEGUNDO: Aplicar offset de ROI si existe
                    if hasattr(self, 'roi_offset'):
                        x += self.roi_offset[0]
                        y += self.roi_offset[1]
                    
                    # TERCERO: Ajustar coordenadas respecto a la bounding box de recorte
                    if bbox_offset:
                        x += bbox_offset[0]  # Sumar offset X de la bounding box
                        y += bbox_offset[1]  # Sumar offset Y de la bounding box
                    
                    coordinates.append([float(x), float(y)])
                
                # Cerrar el polígono
                if len(coordinates) > 0 and coordinates[0] != coordinates[-1]:
                    coordinates.append(coordinates[0])
                
                # Calcular propiedades escaladas correctamente
                area = float(region.area)
                centroid_y, centroid_x = region.centroid
                bbox_min_y, bbox_min_x, bbox_max_y, bbox_max_x = region.bbox
                
                # Escalar propiedades considerando todos los downsamplings
                if total_downsample_factor != 1.0:
                    area *= (total_downsample_factor ** 2)  # Área escala con el cuadrado
                    centroid_x *= total_downsample_factor
                    centroid_y *= total_downsample_factor
                    bbox_min_x *= total_downsample_factor
                    bbox_min_y *= total_downsample_factor
                    bbox_max_x *= total_downsample_factor
                    bbox_max_y *= total_downsample_factor
                
                # Aplicar offset de ROI a propiedades si existe
                if hasattr(self, 'roi_offset'):
                    centroid_x += self.roi_offset[0]
                    centroid_y += self.roi_offset[1]
                    bbox_min_x += self.roi_offset[0]
                    bbox_min_y += self.roi_offset[1]
                    bbox_max_x += self.roi_offset[0]
                    bbox_max_y += self.roi_offset[1]
                
                # Ajustar propiedades respecto a la bounding box de recorte
                if bbox_offset:
                    centroid_x += bbox_offset[0]
                    centroid_y += bbox_offset[1]
                    bbox_min_x += bbox_offset[0]
                    bbox_min_y += bbox_offset[1]
                    bbox_max_x += bbox_offset[0]
                    bbox_max_y += bbox_offset[1]
                
                centroid = [float(centroid_x), float(centroid_y)]
                bbox = [float(bbox_min_x), float(bbox_min_y), float(bbox_max_x), float(bbox_max_y)]
                
                feature = {
                    "type": "Feature",
                    "id": str(uuid.uuid4()),
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coordinates]
                    },
                    "properties": {
                        "classification": {"name": "Positive", "color": [0, 0, 255]},
                        "isLocked": False,
                        "measurements": [],
                        "area": area,
                        "centroid": centroid,
                        "bbox": bbox
                    }
                }
                features.append(feature)
        
        # Preparar metadata incluyendo información completa de downsampling
        metadata = {
            "original_size": getattr(self, 'original_size', None),
            "scaled_size": self.image.shape[::-1] if hasattr(self, 'image') else None,
            "scale_factor": getattr(self, 'scale_factor', 1.0),
            "total_downsample_factor": total_downsample_factor,
            "roi_offset": getattr(self, 'roi_offset', None)
        }
        
        # Agregar información del resultado si está disponible
        if result:
            if 'bounding_box' in result:
                metadata['bounding_box'] = result['bounding_box']
            if 'downsample_info' in result:
                metadata['downsample_info'] = result['downsample_info']
            if 'patch_info' in result:
                metadata['patch_info'] = result['patch_info']
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": metadata
        }

    def _simplify_polygon(self, coordinates, tolerance=2.0):
        """
        Simplifica un polígono usando el algoritmo Douglas-Peucker.
        
        Args:
            coordinates: Lista de coordenadas [x, y]
            tolerance: Tolerancia para la simplificación
            
        Returns:
            list: Coordenadas simplificadas
        """
        if len(coordinates) <= 2:
            return coordinates
        
        # Implementación simple del algoritmo Douglas-Peucker
        def perpendicular_distance(point, line_start, line_end):
            if line_start == line_end:
                return np.linalg.norm(np.array(point) - np.array(line_start))
            
            return abs((line_end[1] - line_start[1]) * point[0] - 
                    (line_end[0] - line_start[0]) * point[1] + 
                    line_end[0] * line_start[1] - 
                    line_end[1] * line_start[0]) / np.linalg.norm(
                        np.array(line_end) - np.array(line_start))

        def douglas_peucker(coords, tolerance):
            if len(coords) <= 2:
                return coords
            
            # Encontrar el punto con la mayor distancia
            dmax = 0
            index = 0
            end = len(coords) - 1
            
            for i in range(1, end):
                d = perpendicular_distance(coords[i], coords[0], coords[end])
                if d > dmax:
                    index = i
                    dmax = d
            
            # Si la distancia máxima es mayor que la tolerancia, dividir recursivamente
            if dmax > tolerance:
                rec_results1 = douglas_peucker(coords[:index+1], tolerance)
                rec_results2 = douglas_peucker(coords[index:], tolerance)
                return rec_results1[:-1] + rec_results2
            else:
                return [coords[0], coords[end]]

        simplified = douglas_peucker(coordinates, tolerance)
        
        # Asegurar que el polígono esté cerrado
        if simplified[0] != simplified[-1]:
            simplified.append(simplified[0])
        
        return simplified
    
    def generate_calibration_report(self) -> Dict:
        """
        Genera un reporte completo de calibración.
        
        Returns:
            dict: Reporte completo de calibración
        """
        self.logger.info("Generando reporte de calibración...")
        
        report = {
            'image_info': {
                'path': str(self.image_path),
                'original_shape': self.image.shape if self.image is not None else None,
                'roi_shape': self.image.shape if self.image is not None else None
            },
            'analysis_results': self.analysis_results
        }
        
        # Guardar reporte en JSON
        with open(self.output_dir / 'calibration_report.json', 'w') as f:
            # Convertir numpy arrays a listas para JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                return obj
            
            import json
            json.dump(report, f, indent=2, default=convert_numpy)
        
        # Crear reporte en texto
        with open(self.output_dir / 'calibration_summary.txt', 'w') as f:
            f.write("=== REPORTE DE CALIBRACIÓN DE PARÁMETROS ===\n\n")
            f.write(f"Imagen analizada: {self.image_path}\n")
            if self.image is not None:
                f.write(f"Tamaño original: {self.image.shape}\n")
            if self.image is not None:
                f.write(f"Tamaño ROI: {self.image.shape}\n")
            f.write("\n")
            
            if 'area_distribution' in self.analysis_results:
                stats = self.analysis_results['area_distribution']['stats']
                f.write("=== DISTRIBUCIÓN DE ÁREAS ===\n")
                f.write(f"Número de regiones: {stats['count']}\n")
                f.write(f"Área media: {stats['mean']:.1f} píxeles\n")
                f.write(f"Área mediana: {stats['median']:.1f} píxeles\n")
                f.write(f"Desviación estándar: {stats['std']:.1f} píxeles\n")
                f.write(f"Rango: {stats['min']:.0f} - {stats['max']:.0f} píxeles\n")
                f.write(f"Área mínima sugerida: {stats['suggested_min_area']} píxeles\n")
                f.write(f"Área máxima sugerida: {stats['suggested_max_area']} píxeles\n\n")
            
            # ...existing code...
            if 'watershed_parameters' in self.analysis_results:
                optimal = self.analysis_results['watershed_parameters']['optimal_params']
                f.write("=== PARÁMETROS ÓPTIMOS DE WATERSHED ===\n")
                f.write(">> Máxima detección:\n")
                f.write(f"   min_distance: {optimal['max_detection']['min_distance']}\n")
                f.write(f"   min_area: {optimal['max_detection']['min_area']}\n")
                f.write(f"   glomérulos esperados: {optimal['max_detection']['expected_count']}\n")
                f.write(f"   área media: {optimal['max_detection']['mean_area']:.1f}\n\n")
                f.write(">> Mejor balance detección/estabilidad:\n")
                f.write(f"   min_distance: {optimal['best_balance']['min_distance']}\n")
                f.write(f"   min_area: {optimal['best_balance']['min_area']}\n")
                f.write(f"   glomérulos esperados: {optimal['best_balance']['expected_count']}\n")
                f.write(f"   área media: {optimal['best_balance']['mean_area']:.1f}\n")
                f.write(f"   stability_score: {optimal['best_balance']['stability_score']:.2f}\n\n")
                f.write(">> Parámetros conservadores:\n")
                f.write(f"   min_distance: {optimal['conservative']['min_distance']}\n")
                f.write(f"   min_area: {optimal['conservative']['min_area']}\n")
                f.write(f"   glomérulos esperados: {optimal['conservative']['expected_count']}\n")
                f.write(f"   área media: {optimal['conservative']['mean_area']:.1f}\n\n")
            f.write("=== FIN DEL REPORTE ===\n")
        return report
# ...existing code...

import os
import glob
from pathlib import Path

def find_glomeruli_composite_files(base_directory):
    """
    Busca archivos que contengan 'glomeruli_composite' en el nombre.
    
    Args:
        base_directory: Directorio base donde buscar
        
    Returns:
        list: Lista de rutas de archivos encontrados
    """
    search_patterns = [
        "**/*glomeruli_composite*.png",
        "**/*glomeruli_composite*.jpg", 
        "**/*glomeruli_composite*.tiff",
        "**/*glomeruli_composite*.tif"
    ]
    
    found_files = []
    base_path = Path(base_directory)
    
    for pattern in search_patterns:
        files = list(base_path.glob(pattern))
        found_files.extend(files)
    
    return found_files

def process_glomeruli_files(base_directory):
    """
    Procesa todos los archivos glomeruli_composite encontrados.
    
    Args:
        base_directory: Directorio base donde buscar
    """
    print(f"🔍 Buscando archivos glomeruli_composite en: {base_directory}")
    
    # Buscar archivos
    glomeruli_files = find_glomeruli_composite_files(base_directory)
    
    if not glomeruli_files:
        print("❌ No se encontraron archivos glomeruli_composite")
        return
    
    print(f"✅ Encontrados {len(glomeruli_files)} archivo(s):")
    for file in glomeruli_files:
        print(f"   - {file}")
    
    # Procesar cada archivo
    for i, image_path in enumerate(glomeruli_files, 1):
        print(f"\n{'='*60}")
        print(f"📊 Procesando archivo {i}/{len(glomeruli_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        # Crear directorio de salida en la misma carpeta del archivo
        output_dir = image_path.parent / "count"
        
        try:
            # Crear instancia del calibrador
            calibrator = GlomeruliParameterCalibrator(str(image_path), str(output_dir))
            
            # Ejecutar análisis
            if calibrator.load_and_prepare_image():
                print("✅ Imagen cargada correctamente")
                
                print("🔍 Analizando operaciones morfológicas...")
                calibrator.analyze_morphological_operations()
                
                print("📊 Analizando distribución de áreas...")
                calibrator.analyze_area_distribution(kernel_size=5)
                
                print("🧪 Probando parámetros de watershed...")
                calibrator.test_watershed_parameters(kernel_size=5)
                
                print("📋 Generando reporte...")
                calibrator.generate_calibration_report()
                
                # Visualizar con parámetros óptimos
                if 'watershed_parameters' in calibrator.analysis_results:
                    optimal_params = calibrator.analysis_results['watershed_parameters']['optimal_params']
                    print("🖼️ Generando visualización con parámetros óptimos...")
                    calibrator.visualize_detection_with_params(
                        min_distance=optimal_params['best_balance']['min_distance'],
                        min_area=optimal_params['best_balance']['min_area']
                    )
                    
                    # Mostrar parámetros encontrados
                    print(f"\n🎯 PARÁMETROS ÓPTIMOS PARA {image_path.name}:")
                    print(f"   Min Distance: {optimal_params['best_balance']['min_distance']}")
                    print(f"   Min Area: {optimal_params['best_balance']['min_area']}")
                    print(f"   Glomérulos esperados: {optimal_params['best_balance']['expected_count']}")
                
                print(f"✅ Análisis completado. Resultados guardados en: {output_dir}")
                
            else:
                print(f"❌ Error al cargar la imagen: {image_path}")
                
        except Exception as e:
            print(f"❌ Error procesando {image_path}: {str(e)}")
            continue
    
    print(f"\n🎉 Procesamiento completado para todos los archivos!")

if __name__ == "__main__":
    # CAMBIA ESTA RUTA por tu directorio
    directory = "/mnt/work/users/bernat.olle/Results/R3/S9"
    
    # Buscar archivo glomeruli_composite.png
    glomeruli_file = None
    for file in Path(directory).glob("*glomeruli_composite*.png"):
        glomeruli_file = file
        break
    
    if glomeruli_file:
        print(f"📁 Encontrado: {glomeruli_file.name}")
        
        # Configurar rutas
        image_path = str(glomeruli_file)
        output_dir = str(glomeruli_file.parent / "count")
        
        # Ejecutar calibración
        calibrator = GlomeruliParameterCalibrator(image_path, output_dir)
        
        if calibrator.load_and_prepare_image():
            calibrator.analyze_morphological_operations()
            calibrator.analyze_area_distribution(kernel_size=5)
            calibrator.test_watershed_parameters(kernel_size=5)
            calibrator.generate_calibration_report()
            
            # Visualizar con parámetros óptimos
            optimal_params = calibrator.analysis_results['watershed_parameters']['optimal_params']
            calibrator.visualize_detection_with_params(
                min_distance=optimal_params['best_balance']['min_distance'],
                min_area=optimal_params['best_balance']['min_area']
            )
            print(f"✅ Completado. Resultados en: {output_dir}")
    else:
        print(f"❌ No se encontró archivo glomeruli_composite.png en: {directory}")