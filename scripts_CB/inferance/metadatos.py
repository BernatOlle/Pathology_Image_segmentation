import os
import re
from datetime import datetime

def parse_log_time(log_line):
    """Extrae el timestamp de una línea de log"""
    time_str = log_line.split(',')[0]
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

def calculate_processing_time(log_file_path):
    """Calcula el tiempo de procesamiento para un archivo de log"""
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return None
        
    start_time = parse_log_time(lines[0])
    end_time = parse_log_time(lines[-1])
    
    return (end_time - start_time).total_seconds()

def analyze_logs(logs_dir):
    """Analiza todos los archivos de log en el directorio especificado"""
    processing_times = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('.log'):
            log_path = os.path.join(logs_dir, filename)
            try:
                time_seconds = calculate_processing_time(log_path)
                if time_seconds is not None:
                    processing_times.append(time_seconds)
                    print(f"{filename}: {time_seconds/60:.2f} minutos")
            except Exception as e:
                print(f"Error procesando {filename}: {str(e)}")
    
    if not processing_times:
        print("No se encontraron archivos de log válidos.")
        return
    
    # Calcular estadísticas
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    
    print("\nEstadísticas:")
    print(f"Tiempo promedio: {avg_time/60:.2f} minutos")
    print(f"Tiempo máximo: {max_time/60:.2f} minutos")
    print(f"Tiempo mínimo: {min_time/60:.2f} minutos")
    print(f"Número de archivos procesados: {len(processing_times)}")

if __name__ == "__main__":
    logs_directory = "logs/R3"
    if os.path.exists(logs_directory):
        analyze_logs(logs_directory)
    else:
        print(f"El directorio {logs_directory} no existe.")