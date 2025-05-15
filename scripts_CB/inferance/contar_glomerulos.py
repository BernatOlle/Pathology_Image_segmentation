import cv2
import numpy as np
from scipy import ndimage
import logging

def count_glomeruli(binary_mask_path=None, binary_mask=None):
    """
    Cuenta los glomérulos en una imagen de máscara binaria compuesta.
    Maneja posibles superposiciones mediante análisis de componentes conectados.
    
    Args:
        binary_mask_path: Ruta a la imagen de máscara binaria compuesta (opcional)
        binary_mask: Array NumPy con la máscara binaria ya cargada (opcional)
        
    Returns:
        int: Número de glomérulos detectados
    """
    
    
    # Configurar logger
    logger = logging.getLogger("glomeruli_counter")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Cargar la máscara si se proporciona la ruta
    if binary_mask is None and binary_mask_path is not None:
        try:
            binary_mask = cv2.imread(str(binary_mask_path), cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                logger.error(f"No se pudo cargar la imagen desde {binary_mask_path}")
                return 0
        except Exception as e:
            logger.error(f"Error al cargar la imagen: {e}")
            return 0
    
    if binary_mask is None:
        logger.error("No se proporcionó ni ruta ni máscara binaria")
        return 0
    
    # Asegurarse de que la máscara sea binaria (0 y 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Aplicar un pequeño kernel de apertura para separar glomérulos levemente conectados
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar componentes conectados
    # El segundo parámetro (4 u 8) determina la conectividad a usar
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask, 8)
    
    # El primer componente (etiqueta 0) es el fondo, así que restamos 1
    num_glomeruli = num_labels - 1
    
    # Filtrar componentes muy pequeños que podrían ser ruido
    min_size = 50  # Ajustar según el tamaño típico de los glomérulos
    valid_glomeruli = 0
    
    for i in range(1, num_labels):  # Empezar desde 1 para omitir el fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            valid_glomeruli += 1
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            logger.info(f"Glomérulo {valid_glomeruli}: Área={area}, Posición=({x},{y}), Tamaño={width}x{height}")
    
    logger.info(f"Total de componentes detectados: {num_labels-1}")
    logger.info(f"Glomérulos válidos (área >= {min_size}): {valid_glomeruli}")
    
    return valid_glomeruli

def analyze_glomeruli_with_watershed(binary_mask_path=None, binary_mask=None, min_distance=20):
    """
    Analiza una imagen de máscara binaria compuesta usando el algoritmo watershed
    para separar glomérulos que podrían estar superpuestos o tocándose.
    
    Args:
        binary_mask_path: Ruta a la imagen de máscara binaria compuesta (opcional)
        binary_mask: Array NumPy con la máscara binaria ya cargada (opcional)
        min_distance: Distancia mínima entre marcadores para la segmentación watershed
        
    Returns:
        tuple: (número de glomérulos, imagen etiquetada)
    """
    import cv2
    import numpy as np
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    import logging
    
    # Configurar logger
    logger = logging.getLogger("glomeruli_analyzer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Cargar la máscara si se proporciona la ruta
    if binary_mask is None and binary_mask_path is not None:
        try:
            binary_mask = cv2.imread(str(binary_mask_path), cv2.IMREAD_GRAYSCALE)
            if binary_mask is None:
                logger.error(f"No se pudo cargar la imagen desde {binary_mask_path}")
                return 0, None
        except Exception as e:
            logger.error(f"Error al cargar la imagen: {e}")
            return 0, None
    
    if binary_mask is None:
        logger.error("No se proporcionó ni ruta ni máscara binaria")
        return 0, None
    
    # Asegurarse de que la máscara sea binaria (0 y 255)
    _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Mejora de la imagen para la segmentación
    # 1. Aplicar un pequeño filtro gaussiano para suavizar bordes
    smoothed = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    
    # 2. Calcular la transformada de distancia
    # Esto nos da la distancia de cada píxel al fondo más cercano
    dist_transform = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)
    
    # 3. Normalizar para mejorar la visualización y procesamiento
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # 4. Encontrar máximos locales en la transformada de distancia
    # Estos serán nuestros marcadores para watershed
    coordinates = peak_local_max(dist_transform, min_distance=min_distance)
    
    # 5. Crear marcadores para watershed
    markers = np.zeros(dist_transform.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coordinates):
        markers[y, x] = i + 1  # Enumerar marcadores desde 1
    
    # 6. Aplicar watershed
    labels = watershed(-dist_transform, markers, mask=binary_mask)
    
    # Contar glomérulos (número de regiones etiquetadas)
    num_glomeruli = len(np.unique(labels)) - 1  # Restar 1 para excluir el fondo (0)
    
    # Analizar propiedades de cada glomérulo
    props = []
    for label in range(1, num_glomeruli + 1):
        # Crear máscara para el glomérulo actual
        glomerulus_mask = (labels == label)
        # Calcular área
        area = np.sum(glomerulus_mask)
        # Encontrar centroide
        indices = np.where(glomerulus_mask)
        if len(indices[0]) > 0:  # Verificar que no esté vacío
            centroid_y = np.mean(indices[0])
            centroid_x = np.mean(indices[1])
            # Guardar propiedades
            props.append({
                'label': label,
                'area': area,
                'centroid': (centroid_x, centroid_y)
            })
            logger.info(f"Glomérulo {label}: Área={area}, Centroide=({centroid_x:.1f}, {centroid_y:.1f})")
    
    logger.info(f"Total de glomérulos detectados con watershed: {num_glomeruli}")
    
    return num_glomeruli, labels

def save_labeled_glomeruli(labels, output_path, colormap=cv2.COLORMAP_JET):
    """
    Guarda una imagen visualizando los glomérulos etiquetados con diferentes colores.
    
    Args:
        labels: Matriz de etiquetas generada por watershed
        output_path: Ruta donde guardar la imagen visualizada
        colormap: Mapa de colores a utilizar
    """
    import cv2
    import numpy as np
    import logging
    
    logger = logging.getLogger("glomeruli_visualizer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    if labels is None:
        logger.error("No se proporcionaron etiquetas para visualizar")
        return
    
    try:
        # Normalizar las etiquetas para visualización
        label_viz = np.zeros(labels.shape, dtype=np.uint8)
        max_label = np.max(labels)
        
        if max_label > 0:  # Verificar que hay etiquetas
            # Escalar a rango 0-255 para visualización
            label_viz = (255 * labels / max_label).astype(np.uint8)
            
            # Aplicar mapa de colores
            label_viz_color = cv2.applyColorMap(label_viz, colormap)
            
            # Hacer que el fondo sea negro (etiqueta 0)
            label_viz_color[labels == 0] = [0, 0, 0]
            
            # Guardar imagen
            cv2.imwrite(str(output_path), label_viz_color)
            logger.info(f"Imagen de glomérulos etiquetados guardada en {output_path}")
        else:
            logger.warning("No se encontraron glomérulos para visualizar")
    
    except Exception as e:
        logger.error(f"Error al guardar imagen etiquetada: {e}")
        import traceback
        traceback.print_exc()

def analyze_glomeruli_complete(binary_mask_path, output_dir=None, min_distance=20):
    """
    Realiza un análisis completo de glomérulos en una imagen de máscara binaria.
    Aplica dos métodos de conteo y guarda una visualización de los resultados.
    
    Args:
        binary_mask_path: Ruta a la imagen de máscara binaria compuesta
        output_dir: Directorio donde guardar los resultados (opcional)
        min_distance: Distancia mínima entre marcadores para watershed
        
    Returns:
        dict: Resultados del análisis
    """
    import os
    import cv2
    import numpy as np
    import logging
    from pathlib import Path
    
    # Configurar logger
    logger = logging.getLogger("glomeruli_analysis")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    logger.info(f"Iniciando análisis de glomérulos en {binary_mask_path}")
    
    # Crear directorio de salida si no existe
    if output_dir is None:
        output_dir = os.path.dirname(binary_mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar la máscara binaria
    try:
        binary_mask = cv2.imread(str(binary_mask_path), cv2.IMREAD_GRAYSCALE)
        if binary_mask is None:
            logger.error(f"No se pudo cargar la imagen desde {binary_mask_path}")
            return {"error": "No se pudo cargar la imagen"}
    except Exception as e:
        logger.error(f"Error al cargar la imagen: {e}")
        return {"error": str(e)}
    
    # Método 1: Conteo simple con componentes conectados
    simple_count = count_glomeruli(binary_mask=binary_mask)
    logger.info(f"Método 1 - Conteo simple: {simple_count} glomérulos")
    
    # Método 2: Análisis con watershed para manejar superposiciones
    watershed_count, labels = analyze_glomeruli_with_watershed(binary_mask=binary_mask, min_distance=min_distance)
    logger.info(f"Método 2 - Watershed: {watershed_count} glomérulos")
    
    # Guardar visualización de glomérulos etiquetados
    if labels is not None:
        labeled_output_path = output_dir / f"{Path(binary_mask_path).stem}_labeled.png"
        save_labeled_glomeruli(labels, labeled_output_path)
    
    # Resultados
    results = {
        "simple_count": simple_count,
        "watershed_count": watershed_count,
        "binary_mask_path": str(binary_mask_path),
        "labeled_output_path": str(labeled_output_path) if labels is not None else None
    }
    
    logger.info(f"Análisis completo. Resultados: {results}")
    return results
  
  
if __name__ == "__main__":
  
      # Ejemplo de uso
      resultado = analyze_glomeruli_complete("/mnt/work/users/bernat.olle/Results/R3/S5/slide-2023-02-18T07-59-52-R3-S5_glomeruli_composite.png")
      print(f"Número de glomérulos detectados: {resultado['watershed_count']}")
      print(f"Imagen etiquetada guardada en: {resultado['labeled_output_path']}")