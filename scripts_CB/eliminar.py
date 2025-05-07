#!/usr/bin/env python3
import os
import sys
import shutil
import time
from pathlib import Path

def mostrar_barra_progreso(actual, total, ancho=50):
    """Muestra una barra de progreso en la terminal."""
    porcentaje = actual / total if total > 0 else 0
    completado = int(ancho * porcentaje)
    barra = '#' * completado + ' ' * (ancho - completado)
    sys.stdout.write(f"\r[{barra}] {int(porcentaje * 100)}% ({actual}/{total})")
    sys.stdout.flush()

def obtener_tamaño_carpeta(ruta):
    """Calcula el tamaño total de una carpeta en bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(ruta):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def formato_tamaño(tamaño_bytes):
    """Formatea un tamaño en bytes a una representación legible."""
    for unidad in ['B', 'KB', 'MB', 'GB', 'TB']:
        if tamaño_bytes < 1024.0:
            return f"{tamaño_bytes:.2f} {unidad}"
        tamaño_bytes /= 1024.0
    return f"{tamaño_bytes:.2f} PB"

def eliminar_carpeta(ruta_carpeta):
    """Elimina una carpeta mostrando el progreso de la operación."""
    ruta = Path(ruta_carpeta)
    
    # Verificar si la carpeta existe
    if not ruta.is_dir():
        print(f"Error: La carpeta {ruta_carpeta} no existe.")
        return False
    
    # Contar archivos para mostrar progreso
    print(f"Contando archivos en {ruta_carpeta}...")
    archivos = []
    for dirpath, dirnames, filenames in os.walk(ruta):
        for f in filenames:
            archivos.append(os.path.join(dirpath, f))
    
    total_archivos = len(archivos)
    print(f"Total de archivos a eliminar: {total_archivos}")
    
    # Calcular y mostrar el espacio ocupado
    tamaño_total = obtener_tamaño_carpeta(ruta_carpeta)
    print(f"Espacio ocupado por la carpeta: {formato_tamaño(tamaño_total)}")
    
    
    
    # Eliminar los archivos uno por uno para mostrar progreso
    print("Eliminando archivos...")
    for i, archivo in enumerate(archivos, 1):
        try:
            os.remove(archivo)
            mostrar_barra_progreso(i, total_archivos)
        except Exception as e:
            print(f"\nError al eliminar {archivo}: {e}")
    
    print("\nEliminando directorios vacíos...")
    try:
        # Eliminar la estructura de directorios
        shutil.rmtree(ruta_carpeta)
        print(f"¡Eliminación completada!")
        print(f"La carpeta {ruta_carpeta} ha sido eliminada correctamente.")
        print(f"Se han liberado {formato_tamaño(tamaño_total)} de espacio en disco.")
        return True
    except Exception as e:
        print(f"Error al eliminar la estructura de directorios: {e}")
        return False

if __name__ == "__main__":
    # Ruta a la carpeta que se va a eliminar
    carpeta_a_eliminar = "/mnt/work/users/bernat.olle/Dataset/validation/S22"
    
    # Si se proporciona un argumento, usar esa ruta en su lugar
    if len(sys.argv) > 1:
        carpeta_a_eliminar = sys.argv[1]
    
    # Iniciar el proceso de eliminación
    eliminar_carpeta(carpeta_a_eliminar)