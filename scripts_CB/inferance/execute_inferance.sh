#!/bin/bash

# Directorio donde se encuentran las imágenes MRXS
INPUT_DIR="/mnt/work/datasets/BKidney/CROC/"

echo "Buscando archivos MRXS con 'R5' en su nombre..."

# Encontrar todos los archivos .mrxs que contienen "R3" en su nombre
mrxs_files=$(find "$INPUT_DIR" -name "*.mrxs" | grep -E "R5|R4")


# Verificar si se encontraron archivos
if [ -z "$mrxs_files" ]; then
    echo "No se encontraron archivos .mrxs que contengan 'R5' en el directorio $INPUT_DIR"
    exit 1
fi

# Convertir la salida en un array para poder procesarlo más fácilmente
readarray -t mrxs_array <<< "$mrxs_files"

echo "Se encontraron ${#mrxs_array[@]} archivos."
echo "Iniciando procesamiento en grupos de 5..."

# Contador para identificar los trabajos
count=1
# Contador para controlar grupos de 5
group_count=0
# Número de trabajos en paralelo
max_parallel=10

for mrxs_file in "${mrxs_array[@]}"; do
    # Extraer el nombre del archivo
    filename=$(basename "$mrxs_file")
    
    echo "Lanzando trabajo para $filename"
    
    # Ejecutar srun en segundo plano sin redirección a log
    srun --gres=gpu:1 --mem=32G --cpus-per-task=8 --time=02:00:00 \
         python parcher_mrxs.py --input_dir "$mrxs_file" --patch_size 2048 --save_geojson --skip_background &
    
    # Guardar el PID del proceso
    pid=$!
    echo "Trabajo #$count lanzado con PID $pid"
    
    # Incrementar contadores
    ((count++))
    ((group_count++))
    
    # Si hemos lanzado 5 trabajos, esperar a que terminen antes de continuar
    if [ $group_count -eq $max_parallel ]; then
        echo "Esperando a que terminen los 5 trabajos actuales..."
        wait
        echo "Grupo completado. Continuando con el siguiente grupo..."
        group_count=0
    fi
done

# Esperar a que terminen los trabajos restantes (si hay menos de 5 en el último grupo)
if [ $group_count -gt 0 ]; then
    echo "Esperando a que terminen los $group_count trabajos restantes..."
    wait
    echo "Grupo final completado."
fi

echo "Todos los trabajos han finalizado. Se procesaron un total de $((count-1)) archivos."