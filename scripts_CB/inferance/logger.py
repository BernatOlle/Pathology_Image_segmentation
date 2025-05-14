import logging
import os
import glob

# Variable global para el logger
_logger = None

def setup_logger(input_dir, logs_dir=None, console_output=False):
    """
    Configura un logger global usando el nombre del archivo MRXS con sufijo '_logs'
    
    Args:
        input_dir: Ruta al archivo MRXS o directorio que contiene archivos MRXS
        logs_dir: Directorio donde se guardarán los archivos de log (opcional)
        console_output: Si es True, también muestra logs en la consola (por defecto: False)
    """
    global _logger
    
    # Si el logger ya está configurado, simplemente devuélvelo
    if _logger is not None:
        return _logger
    
    # Determinar si el input_dir es directamente un archivo MRXS o un directorio
    if os.path.isfile(input_dir) and input_dir.lower().endswith('.mrxs'):
        # Si el input_dir es directamente un archivo MRXS
        mrxs_path = input_dir
    else:
        # Buscar un archivo MRXS en el directorio
        mrxs_files = glob.glob(os.path.join(input_dir, '*.mrxs'))
        if mrxs_files:
            mrxs_path = mrxs_files[0]  # Tomar el primer archivo MRXS encontrado
        else:
            # Si no hay archivos MRXS, usar un nombre por defecto
            _logger = logging.getLogger('global_logger')
            return _logger  # Logger vacío como fallback
    
    # Obtener el nombre base del archivo MRXS para el log
    mrxs_basename = os.path.basename(mrxs_path)
    mrxs_name = os.path.splitext(mrxs_basename)[0]
    
    # Si no se especifica una carpeta de logs, usar './logs'
    if logs_dir is None:
        logs_dir = os.path.join(os.getcwd(), 'logs')
    
    # Crear el directorio para logs si no existe
    os.makedirs(logs_dir, exist_ok=True)
    
    # Ruta completa del archivo de log
    log_filename = os.path.join(logs_dir, f"{mrxs_name}_logs.log")
    
    # Configurar el logger global
    _logger = logging.getLogger('global_logger')
    _logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes para evitar duplicados
    if _logger.handlers:
        for handler in _logger.handlers:
            _logger.removeHandler(handler)
    
    # Crear formato con más detalles
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para consola (solo si console_output es True)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)
    
    # Handler para archivo (siempre se añade)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    
    _logger.info(f"Iniciando procesamiento del archivo: {mrxs_path}")
    _logger.info(f"Log guardado en: {log_filename}")
    
    return _logger

def get_logger():
    """
    Obtiene el logger global. Si no está configurado, devuelve un logger básico.
    """
    global _logger
    if _logger is None:
        # Configurar un logger básico si aún no se ha inicializado
        _logger = logging.getLogger('global_logger')
        _logger.setLevel(logging.INFO)
        
        # Verificar si ya tiene handlers para evitar duplicados
        if not _logger.handlers:
            # Solo crear un handler de archivo, no de consola
            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            log_filename = os.path.join(logs_dir, "default_logs.log")
            
            file_handler = logging.FileHandler(log_filename)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
            
            # Esto se escribe en el archivo, no en la consola
            _logger.warning("Logger no inicializado correctamente. Usando configuración por defecto.")
    
    return _logger