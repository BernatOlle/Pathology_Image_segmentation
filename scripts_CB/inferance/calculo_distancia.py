import json
import math
from typing import List, Tuple, Dict

def load_geojson(file_path: str) -> dict:
    """Carga el archivo GeoJSON"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_polygon_coordinates(feature: dict) -> List[List[float]]:
    """Extrae las coordenadas del polígono de una feature"""
    geometry = feature['geometry']
    if geometry['type'] == 'Polygon':
        # Tomamos solo el anillo exterior (primer elemento)
        return geometry['coordinates'][0]
    return []

def calculate_bounding_box(coordinates: List[List[float]]) -> Tuple[float, float, float, float]:
    """Calcula el bounding box (min_x, min_y, max_x, max_y) de un conjunto de coordenadas"""
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def calculate_max_distances(coordinates: List[List[float]]) -> Dict[str, float]:
    """Calcula las distancias máximas horizontales y verticales"""
    if len(coordinates) < 2:
        return {'max_horizontal': 0, 'max_vertical': 0}
    
    # Método 1: Usando bounding box (más eficiente)
    min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)
    horizontal_distance = max_x - min_x
    vertical_distance = max_y - min_y
    
    # Método 2: Calculando todas las distancias punto a punto (más preciso)
    max_horizontal_distance = 0
    max_vertical_distance = 0
    
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            x1, y1 = coordinates[i][0], coordinates[i][1]
            x2, y2 = coordinates[j][0], coordinates[j][1]
            
            # Distancia horizontal (diferencia en X)
            horizontal_dist = abs(x2 - x1)
            if horizontal_dist > max_horizontal_distance:
                max_horizontal_distance = horizontal_dist
            
            # Distancia vertical (diferencia en Y)
            vertical_dist = abs(y2 - y1)
            if vertical_dist > max_vertical_distance:
                max_vertical_distance = vertical_dist
    
    return {
        'max_horizontal_bbox': horizontal_distance,
        'max_vertical_bbox': vertical_distance,
        'max_horizontal_points': max_horizontal_distance,
        'max_vertical_points': max_vertical_distance
    }

def calculate_euclidean_distance(coord1: List[float], coord2: List[float]) -> float:
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

def calculate_max_euclidean_distance(coordinates: List[List[float]]) -> float:
    """Calcula la distancia euclidiana máxima entre todos los puntos del polígono"""
    max_distance = 0
    
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = calculate_euclidean_distance(coordinates[i], coordinates[j])
            if distance > max_distance:
                max_distance = distance
    
    return max_distance

def analyze_geojson_distances(geojson_data: dict) -> Dict:
    """Analiza todas las distancias del GeoJSON"""
    results = {
        'polygon_count': 0,
        'polygons': [],
        'statistics': {
            'max_horizontal_overall': 0,
            'max_vertical_overall': 0,
            'max_euclidean_overall': 0,
            'avg_horizontal': 0,
            'avg_vertical': 0,
            'avg_euclidean': 0
        }
    }
    
    horizontal_distances = []
    vertical_distances = []
    euclidean_distances = []
    
    for i, feature in enumerate(geojson_data['features']):
        coordinates = get_polygon_coordinates(feature)
        
        if not coordinates:
            continue
        
        distances = calculate_max_distances(coordinates)
        max_euclidean = calculate_max_euclidean_distance(coordinates)
        
        polygon_result = {
            'id': feature.get('id', f'polygon_{i}'),
            'coordinates_count': len(coordinates),
            'max_horizontal_bbox': distances['max_horizontal_bbox'],
            'max_vertical_bbox': distances['max_vertical_bbox'],
            'max_horizontal_points': distances['max_horizontal_points'],
            'max_vertical_points': distances['max_vertical_points'],
            'max_euclidean_distance': max_euclidean
        }
        
        results['polygons'].append(polygon_result)
        
        # Recolectar para estadísticas (usando los valores de puntos más precisos)
        horizontal_distances.append(distances['max_horizontal_points'])
        vertical_distances.append(distances['max_vertical_points'])
        euclidean_distances.append(max_euclidean)
    
    results['polygon_count'] = len(results['polygons'])
    
    if horizontal_distances:
        results['statistics']['max_horizontal_overall'] = max(horizontal_distances)
        results['statistics']['max_vertical_overall'] = max(vertical_distances)
        results['statistics']['max_euclidean_overall'] = max(euclidean_distances)
        results['statistics']['avg_horizontal'] = sum(horizontal_distances) / len(horizontal_distances)
        results['statistics']['avg_vertical'] = sum(vertical_distances) / len(vertical_distances)
        results['statistics']['avg_euclidean'] = sum(euclidean_distances) / len(euclidean_distances)
    
    return results

def print_results(results: Dict):
    """Imprime los resultados de forma organizada"""
    print("="*60)
    print("ANÁLISIS DE DISTANCIAS EN GEOJSON")
    print("="*60)
    
    print(f"\nTotal de polígonos analizados: {results['polygon_count']}")
    
    print("\n" + "="*40)
    print("ESTADÍSTICAS GENERALES")
    print("="*40)
    
    stats = results['statistics']
    print(f"Distancia horizontal máxima: {stats['max_horizontal_overall']:.2f}")
    print(f"Distancia vertical máxima: {stats['max_vertical_overall']:.2f}")
    print(f"Distancia euclidiana máxima: {stats['max_euclidean_overall']:.2f}")
    print(f"")
    print(f"Distancia horizontal promedio: {stats['avg_horizontal']:.2f}")
    print(f"Distancia vertical promedio: {stats['avg_vertical']:.2f}")
    print(f"Distancia euclidiana promedio: {stats['avg_euclidean']:.2f}")
    
    # Determinar cuál es la dimensión máxima
    max_dimension = max(stats['max_horizontal_overall'], stats['max_vertical_overall'])
    dimension_type = "horizontal" if stats['max_horizontal_overall'] > stats['max_vertical_overall'] else "vertical"
    
    print(f"\n🎯 RESPUESTA A TU PREGUNTA:")
    print(f"La longitud máxima es {max_dimension:.2f} y está en dirección {dimension_type}")
    
    print("\n" + "="*40)
    

# Función principal
def main():
    # Cambia esta ruta por la ruta de tu archivo
    file_path = "/mnt/work/users/bernat.olle/Results/R3/S20/geojson/slide-2023-02-18T08-21-34-R3-S20_glomeruli.geojson_consolidated.geojson"
    
    try:
        # Cargar el GeoJSON
        geojson_data = load_geojson(file_path)
        
        # Analizar distancias
        results = analyze_geojson_distances(geojson_data)
        
        # Mostrar resultados
        print_results(results)
        
        # Opcional: guardar resultados en JSON
        with open('resultados_distancias.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Resultados guardados en 'resultados_distancias.json'")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {file_path}")
        print("Por favor, asegúrate de que el archivo existe y la ruta es correcta.")
    except json.JSONDecodeError:
        print("❌ Error: El archivo no tiene un formato JSON válido.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

# Si quieres probar con datos de ejemplo (tu polígono)


if __name__ == "__main__":
 
    
    # Para usar con tu archivo real:
    main()