#!/usr/bin/env python3
"""
Analizador de área blanca en glomérulos de riñones de rata
Función reutilizable para medir el área blanca entre el núcleo y el perímetro del glomérulo
"""

import json
import numpy as np
import cv2
import os
from pathlib import Path
import openslide
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class GlomeruliWhiteAreaAnalyzer:
    def __init__(self, slide_path, geojson_path, output_dir, min_area_pixels=7, mask_expansion_pixels=50):
        self.slide_path = slide_path
        self.geojson_path = geojson_path
        self.output_dir = Path(output_dir)
        self.slide = None
        self.geojson_data = None
        self.metadata = None
        
        self.min_area_pixels = min_area_pixels
        self.mask_expansion_pixels = mask_expansion_pixels
        
        # Bounds information
        self.bounds_x = 0
        self.bounds_y = 0
        self.bounds_width = 0
        self.bounds_height = 0
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_slide(self):
        """Cargar la whole slide image y extraer metadatos"""
        try:
            self.slide = openslide.OpenSlide(self.slide_path)
            
            full_width = self.slide.dimensions[0]
            full_height = self.slide.dimensions[1]
            
            self.bounds_x = int(self.slide.properties.get('openslide.bounds-x', 0))
            self.bounds_y = int(self.slide.properties.get('openslide.bounds-y', 0))
            self.bounds_width = int(self.slide.properties.get('openslide.bounds-width', full_width))
            self.bounds_height = int(self.slide.properties.get('openslide.bounds-height', full_height))
            
            return True
        except Exception:
            return False
    
    def load_geojson(self):
        """Cargar el archivo GeoJSON con las máscaras de glomérulos"""
        try:
            with open(self.geojson_path, 'r') as f:
                self.geojson_data = json.load(f)
            
            if 'metadata' in self.geojson_data:
                self.metadata = self.geojson_data['metadata']
            
            return True
        except Exception:
            return False
    
    def get_glomerulus_region(self, feature, padding=50):
        """Extraer la región del glomérulo de la slide considerando los bounds offset"""
        coords = feature['geometry']['coordinates'][0]
        polygon = Polygon(coords)
        
        minx, miny, maxx, maxy = polygon.bounds
        
        minx -= padding
        miny -= padding
        maxx += padding
        maxy += padding
        
        minx_absolute = int(minx) + self.bounds_x
        miny_absolute = int(miny) + self.bounds_y
        maxx_absolute = int(maxx) + self.bounds_x
        maxy_absolute = int(maxy) + self.bounds_y
        
        if self.metadata and False:
            scale_factor = self.metadata.get('scale_factor', 1.0)
            total_downsample = self.metadata.get('total_downsample_factor', 1.0)
            
            minx_absolute = int(minx_absolute / scale_factor * total_downsample)
            miny_absolute = int(miny_absolute / scale_factor * total_downsample)
            maxx_absolute = int(maxx_absolute / scale_factor * total_downsample)
            maxy_absolute = int(maxy_absolute / scale_factor * total_downsample)
        
        width, height = self.slide.dimensions
        minx_absolute = max(0, minx_absolute)
        miny_absolute = max(0, miny_absolute)
        maxx_absolute = min(width, maxx_absolute)
        maxy_absolute = min(height, maxy_absolute)
        
        region_width = maxx_absolute - minx_absolute
        region_height = maxy_absolute - miny_absolute
        
        if region_width <= 0 or region_height <= 0:
            return None, None, None
        
        region = self.slide.read_region((minx_absolute, miny_absolute), 0, (region_width, region_height))
        region_rgb = np.array(region.convert('RGB'))
        
        return region_rgb, (minx_absolute, miny_absolute, maxx_absolute, maxy_absolute), polygon
    
    def create_glomerulus_mask(self, region_shape, polygon, bbox):
        """Crear máscara del glomérulo en la región extraída considerando bounds offset"""
        minx_absolute, miny_absolute, maxx_absolute, maxy_absolute = bbox
        
        coords = list(polygon.exterior.coords)
        
        if self.metadata and False:
            scale_factor = self.metadata.get('scale_factor', 1.0)
            total_downsample = self.metadata.get('total_downsample_factor', 1.0)
            scaling = total_downsample / scale_factor
        else:
            scaling = 1.0
        
        local_coords = []
        for x, y in coords:
            abs_x = (x + self.bounds_x) * scaling
            abs_y = (y + self.bounds_y) * scaling
            local_x = int(abs_x - minx_absolute)
            local_y = int(abs_y - miny_absolute)
            local_coords.append([local_x, local_y])
        
        mask_basic = np.zeros(region_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_basic, [np.array(local_coords, dtype=np.int32)], 255)
        
        if self.mask_expansion_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.mask_expansion_pixels * 2, self.mask_expansion_pixels * 2))
            mask_expanded = cv2.dilate(mask_basic, kernel, iterations=1)
            return mask_expanded, local_coords
        else:
            return mask_basic, local_coords
    
    def segment_white_area(self, region, glomerulus_mask):
        """Segmentar el área blanca dentro del glomérulo usando técnicas de computer vision"""
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([180, 40, 255])
        
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        
        white_in_glomerulus = cv2.bitwise_and(white_mask, glomerulus_mask)
        
        white_filtered = self.filter_by_area(white_in_glomerulus, min_area=self.min_area_pixels)
        
        total_white_area = np.sum(white_filtered == 255)
        if total_white_area > 100:
            white_refined = self.apply_watershed(region, white_filtered, glomerulus_mask)
        else:
            white_refined = white_filtered
        
        return white_refined, white_mask
    
    def filter_by_area(self, binary_mask, min_area=5):
        """Filtrar componentes conectados por área mínima"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        filtered_mask = np.zeros_like(binary_mask)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered_mask[labels == i] = 255
        
        return filtered_mask
    
    def apply_watershed(self, region, white_mask, glomerulus_mask):
        """Aplicar algoritmo watershed para refinar la segmentación del área blanca"""
        background = cv2.dilate(glomerulus_mask, np.ones((3,3), np.uint8), iterations=2)
        background = cv2.bitwise_not(background)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        foreground = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        foreground = cv2.erode(foreground, kernel, iterations=1)
        
        unknown = cv2.subtract(white_mask, foreground)
        
        _, markers = cv2.connectedComponents(foreground)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), markers)
        
        refined_mask = np.zeros_like(white_mask)
        refined_mask[markers > 1] = 255
        
        refined_mask = cv2.bitwise_and(refined_mask, glomerulus_mask)
        
        refined_mask = self.filter_by_area(refined_mask, min_area=self.min_area_pixels)
        
        return refined_mask
    
    def calculate_white_area(self, white_mask):
        """Calcular el área blanca en píxeles y convertir a unidades reales si es posible"""
        white_pixels = np.sum(white_mask == 255)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
        num_components = num_labels - 1
        
        component_areas = []
        if num_components > 0:
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                component_areas.append(int(area))
        
        try:
            mpp_x = float(self.slide.properties.get('openslide.mpp-x', 1.0))
            mpp_y = float(self.slide.properties.get('openslide.mpp-y', 1.0))
            
            area_um2 = float(white_pixels * mpp_x * mpp_y)
            return int(white_pixels), area_um2, int(num_components), component_areas
        except:
            return int(white_pixels), None, int(num_components), component_areas
    
    def visualize_results(self, region, glomerulus_mask, white_mask, local_coords, glomerulus_id):
        """Crear visualización de los resultados"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(region)
        axes[0, 0].set_title('Región Original del Glomérulo')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(glomerulus_mask, cmap='gray')
        axes[0, 1].set_title(f'Máscara del Glomérulo\n(Expandida +{self.mask_expansion_pixels}px)')
        if local_coords:
            coords_array = np.array(local_coords)
            axes[0, 1].plot(coords_array[:, 0], coords_array[:, 1], 'b-', linewidth=2, label='Contorno Original')
            axes[0, 1].legend()
        axes[0, 1].axis('off')
        
        mask_original = np.zeros(region.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_original, [np.array(local_coords, dtype=np.int32)], 128)
        mask_expansion = glomerulus_mask.copy()
        mask_expansion[mask_original == 128] = 255
        
        axes[0, 2].imshow(mask_expansion, cmap='RdYlBu_r')
        axes[0, 2].set_title('Máscara: Original (azul) + Expansión (rojo)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(white_mask, cmap='gray')
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            axes[1, 0].text(cx, cy, f'{i}\n({area}px)', ha='center', va='center', 
                           color='red', fontsize=8, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        axes[1, 0].set_title(f'Área Blanca Segmentada\n({num_labels-1} componentes)')
        axes[1, 0].axis('off')
        
        overlay1 = region.copy()
        overlay1[white_mask == 255] = [0, 255, 80]
        if local_coords:
            coords_array = np.array(local_coords)
            axes[1, 1].imshow(overlay1)
            axes[1, 1].plot(coords_array[:, 0], coords_array[:, 1], 'b-', linewidth=2, label='Contorno Original')
            axes[1, 1].set_title('Área Blanca + Contorno Original')
            axes[1, 1].legend()
            axes[1, 1].axis('off')
        
        overlay2 = region.copy()
        overlay2[white_mask == 255] = [0, 255, 80]
        mask_border = cv2.Canny(glomerulus_mask, 50, 150)
        overlay2[mask_border == 255] = [0, 255, 0]
        
        axes[1, 2].imshow(overlay2)
        axes[1, 2].set_title('Área Blanca + Borde Expandido (verde)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f'glomerulus_{glomerulus_id}_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def process_single_glomerulus(self, feature, glomerulus_id):
        """Procesar un solo glomérulo"""
        region, bbox, polygon = self.get_glomerulus_region(feature)
        if region is None:
            return None
        
        glomerulus_mask, local_coords = self.create_glomerulus_mask(region.shape, polygon, bbox)
        
        white_mask, raw_white_mask = self.segment_white_area(region, glomerulus_mask)
        
        white_pixels, white_area_um2, num_components, component_areas = self.calculate_white_area(white_mask)
        
        viz_path = self.visualize_results(region, glomerulus_mask, white_mask, local_coords, glomerulus_id)
        
        results = {
            'glomerulus_id': int(glomerulus_id),
            'white_area_pixels': int(white_pixels),
            'white_area_um2': float(white_area_um2) if white_area_um2 is not None else None,
            'num_components': int(num_components),
            'component_areas': [int(area) for area in component_areas],
            'min_component_area': int(min(component_areas)) if component_areas else 0,
            'max_component_area': int(max(component_areas)) if component_areas else 0,
            'avg_component_area': float(np.mean(component_areas)) if component_areas else 0.0,
            'bbox': [int(x) for x in bbox],
            'visualization_path': str(viz_path)
        }
        
        return results
    
    def process_all_glomeruli(self):
        """Procesar todos los glomérulos en el GeoJSON"""
        if not self.load_slide() or not self.load_geojson():
            return None
        
        results = []
        
        for i, feature in enumerate(self.geojson_data['features']):
            try:
                result = self.process_single_glomerulus(feature, i)
                if result:
                    results.append(result)
            except Exception:
                continue
        
        results_data = {
            'slide_path': self.slide_path,
            'slide_name': Path(self.slide_path).stem,
            'geojson_path': self.geojson_path,
            'metadata': self.metadata,
            'analysis_parameters': {
                'min_area_pixels': self.min_area_pixels,
                'mask_expansion_pixels': self.mask_expansion_pixels,
                'white_hsv_range': {
                    'lower': [0, 0, 160],
                    'upper': [180, 40, 255]
                }
            },
            'bounds_info': {
                'bounds_x': self.bounds_x,
                'bounds_y': self.bounds_y,
                'bounds_width': self.bounds_width,
                'bounds_height': self.bounds_height
            },
            'total_glomeruli': len(results),
            'results': results
        }
        
        results_path = self.output_dir / 'white_area_analysis.json'
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.create_summary_report(results_data)
        
        return results_data
    
    def create_summary_report(self, results_data):
        """Crear un reporte resumen de los resultados"""
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("REPORTE DE ANÁLISIS DE ÁREA BLANCA EN GLOMÉRULOS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Slide: {results_data['slide_path']}\n")
            f.write(f"Slide name: {results_data['slide_name']}\n")
            f.write(f"GeoJSON: {results_data['geojson_path']}\n")
            f.write(f"Total de glomérulos procesados: {results_data['total_glomeruli']}\n")
            f.write(f"Área mínima configurada: {results_data['analysis_parameters']['min_area_pixels']} píxeles\n")
            f.write(f"Expansión de máscara: {results_data['analysis_parameters']['mask_expansion_pixels']} píxeles\n\n")
            
            f.write("BOUNDS INFORMATION:\n")
            bounds_info = results_data['bounds_info']
            f.write(f"  Bounds position: ({bounds_info['bounds_x']}, {bounds_info['bounds_y']})\n")
            f.write(f"  Bounds size: {bounds_info['bounds_width']} x {bounds_info['bounds_height']}\n\n")
            
            if results_data['metadata']:
                f.write("METADATA:\n")
                for key, value in results_data['metadata'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("PARÁMETROS DE ANÁLISIS:\n")
            params = results_data['analysis_parameters']
            f.write(f"  Área mínima: {params['min_area_pixels']} píxeles\n")
            f.write(f"  Expansión de máscara: {params['mask_expansion_pixels']} píxeles\n")
            hsv_range = params['white_hsv_range']
            f.write(f"  Rango HSV: {hsv_range['lower']} - {hsv_range['upper']}\n\n")
            
            f.write("RESULTADOS POR GLOMÉRULO:\n")
            f.write("-" * 30 + "\n")
            
            total_white_pixels = 0
            total_white_area_um2 = 0
            total_components = 0
            all_component_sizes = []
            
            for result in results_data['results']:
                f.write(f"Glomérulo {result['glomerulus_id']}:\n")
                f.write(f"  Área blanca: {result['white_area_pixels']} píxeles\n")
                f.write(f"  Componentes: {result['num_components']}\n")
                if result['component_areas']:
                    f.write(f"  Tamaños: {result['min_component_area']}-{result['max_component_area']} píxeles (prom: {result['avg_component_area']:.1f})\n")
                    all_component_sizes.extend(result['component_areas'])
                if result['white_area_um2']:
                    f.write(f"  Área blanca: {result['white_area_um2']:.2f} μm²\n")
                    total_white_area_um2 += result['white_area_um2']
                f.write(f"  Visualización: {result['visualization_path']}\n\n")
                
                total_white_pixels += result['white_area_pixels']
                total_components += result['num_components']
            
            f.write("RESUMEN TOTAL:\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total área blanca: {total_white_pixels} píxeles\n")
            f.write(f"Total componentes detectados: {total_components}\n")
            if all_component_sizes:
                f.write(f"Tamaño componentes: {min(all_component_sizes)}-{max(all_component_sizes)} píxeles\n")
                f.write(f"Tamaño promedio componente: {np.mean(all_component_sizes):.1f} píxeles\n")
            if total_white_area_um2 > 0:
                f.write(f"Total área blanca: {total_white_area_um2:.2f} μm²\n")
                f.write(f"Promedio por glomérulo: {total_white_area_um2/len(results_data['results']):.2f} μm²\n")
                f.write(f"Promedio por componente: {total_white_area_um2/total_components:.2f} μm²\n")


# FUNCIÓN PRINCIPAL PARA USO EXTERNO
def analyze_glomeruli_white_area(slide_path, geojson_path, output_dir, 
                                min_area_pixels=7, mask_expansion_pixels=50):
    """
    Función principal para analizar área blanca en glomérulos
    
    Parámetros:
    -----------
    slide_path : str
        Ruta al archivo .mrxs de la slide
    geojson_path : str  
        Ruta al archivo GeoJSON con las máscaras de glomérulos
    output_dir : str
        Directorio donde guardar los resultados
    min_area_pixels : int, opcional (default=7)
        Área mínima en píxeles para detectar componentes
    mask_expansion_pixels : int, opcional (default=50)
        Píxeles para expandir la máscara del glomérulo
        
    Retorna:
    --------
    dict o None
        Diccionario con los resultados del análisis o None si hay error
    """
    
    # Verificar que los archivos existan
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"No se encuentra el archivo slide: {slide_path}")
    
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"No se encuentra el archivo GeoJSON: {geojson_path}")
    
    # Crear analizador y procesar
    analyzer = GlomeruliWhiteAreaAnalyzer(
        slide_path, 
        geojson_path, 
        output_dir, 
        min_area_pixels=min_area_pixels,
        mask_expansion_pixels=mask_expansion_pixels
    )
    
    results = analyzer.process_all_glomeruli()
    
    return results


# Función simplificada para un solo glomérulo
def analyze_single_glomerulus(slide_path, geojson_path, glomerulus_index, output_dir,
                             min_area_pixels=7, mask_expansion_pixels=50):
    """
    Analizar un solo glomérulo específico
    
    Parámetros:
    -----------
    slide_path : str
        Ruta al archivo .mrxs de la slide
    geojson_path : str  
        Ruta al archivo GeoJSON con las máscaras de glomérulos
    glomerulus_index : int
        Índice del glomérulo a procesar (empezando desde 0)
    output_dir : str
        Directorio donde guardar los resultados
    min_area_pixels : int, opcional (default=7)
        Área mínima en píxeles para detectar componentes
    mask_expansion_pixels : int, opcional (default=50)
        Píxeles para expandir la máscara del glomérulo
        
    Retorna:
    --------
    dict o None
        Diccionario con los resultados del análisis del glomérulo específico
    """
    
    analyzer = GlomeruliWhiteAreaAnalyzer(
        slide_path, 
        geojson_path, 
        output_dir, 
        min_area_pixels=min_area_pixels,
        mask_expansion_pixels=mask_expansion_pixels
    )
    
    if not analyzer.load_slide() or not analyzer.load_geojson():
        return None
    
    if glomerulus_index >= len(analyzer.geojson_data['features']):
        raise IndexError(f"Índice de glomérulo {glomerulus_index} fuera de rango. Total: {len(analyzer.geojson_data['features'])}")
    
    feature = analyzer.geojson_data['features'][glomerulus_index]
    result = analyzer.process_single_glomerulus(feature, glomerulus_index)
    
    return result