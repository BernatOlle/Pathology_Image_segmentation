#!/usr/bin/env python
import os
import re
import glob
from datetime import datetime, timedelta

def extract_processing_time(log_file):
    """
    Extrae el tiempo total de procesamiento de un archivo de registro.
    Busca la última ocurrencia de 'INFO - Tiempo total de procesamiento: X:XX:XX'
    """
    processing_time = None
    
    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Buscar todas las ocurrencias del patrón de tiempo de procesamiento
            time_matches = re.findall(r'INFO - Tiempo total de procesamiento: (\d+:\d+:\d+)', content)
            
            if time_matches:
                # Tomar el último tiempo encontrado
                processing_time = time_matches[-1]
    except Exception as e:
        print(f"Error al procesar el archivo {log_file}: {e}")
    
    return processing_time

def time_str_to_seconds(time_str):
    """Convierte una cadena de tiempo (H:M:S) a segundos totales"""
    if not time_str:
        return 0
    
    parts = time_str.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    else:
        return int(parts[0])

def seconds_to_time_str(seconds):
    """Convierte segundos a formato H:M:S"""
    return str(timedelta(seconds=seconds))

def analyze_log_folder(log_folder_path):
    """
    Analiza todos los archivos de registro en una carpeta y calcula el tiempo promedio de procesamiento
    """
    if not os.path.exists(log_folder_path):
        print(f"Error: La carpeta {log_folder_path} no existe.")
        return
    
    log_files = glob.glob(os.path.join(log_folder_path, "*.log"))
    
    if not log_files:
        print(f"No se encontraron archivos de registro en {log_folder_path}")
        return
    
    total_seconds = 0
    processed_files = 0
    processing_times = []
    
    print(f"Analizando {len(log_files)} archivos de registro...")
    
    for log_file in log_files:
        time_str = extract_processing_time(log_file)
        
        if time_str:
            seconds = time_str_to_seconds(time_str)
            total_seconds += seconds
            processed_files += 1
            processing_times.append((os.path.basename(log_file), time_str, seconds))
            print(f"Archivo: {os.path.basename(log_file)} - Tiempo: {time_str}")
        else:
            print(f"No se encontró información de tiempo en {os.path.basename(log_file)}")
    
    if processed_files > 0:
        avg_seconds = total_seconds / processed_files
        avg_time_str = seconds_to_time_str(avg_seconds)
        
        print("\n--- Resultados ---")
        print(f"Total de archivos procesados: {processed_files}")
        print(f"Tiempo total acumulado: {seconds_to_time_str(total_seconds)}")
        print(f"Tiempo promedio de procesamiento: {avg_time_str}")
        
        # Identificar tiempos mínimo y máximo
        if processing_times:
            min_time = min(processing_times, key=lambda x: x[2])
            max_time = max(processing_times, key=lambda x: x[2])
            
            print(f"\nTiempo mínimo: {min_time[1]} ({min_time[0]})")
            print(f"Tiempo máximo: {max_time[1]} ({max_time[0]})")
    else:
        print("No se encontró información de tiempo en ningún archivo.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analiza los tiempos de procesamiento en archivos de registro.')
    parser.add_argument('--folder', type=str, default='logs', 
                        help='Ruta a la carpeta que contiene los archivos de registro')
    
    args = parser.parse_args()
    
    analyze_log_folder(args.folder)