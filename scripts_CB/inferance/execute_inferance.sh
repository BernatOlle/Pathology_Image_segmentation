#!/bin/bash

# Directorio donde se encuentran las imágenes MRXS
INPUT_DIR="/mnt/work/datasets/BKidney/CROC/"

echo "Buscando archivos MRXS con 'R3' en su nombre..."

# Encontrar todos los archivos .mrxs que contienen "R3" en su nombre
mrxs_files=$(find "$INPUT_DIR" -name "*.mrxs" | grep -E "R3")

# Verificar si se encontraron archivos
if [ -z "$mrxs_files" ]; then
    echo "No se encontraron archivos .mrxs que contengan 'R3' en el directorio $INPUT_DIR"
    exit 1
fi

# Convertir la salida en un array para poder procesarlo más fácilmente
readarray -t mrxs_array <<< "$mrxs_files"

echo "Se encontraron ${#mrxs_array[@]} archivos."
echo "Iniciando procesamiento con máximo 10 procesos en paralelo..."

# Número máximo de trabajos en paralelo
max_parallel=20

# Arrays para mantener el control de los procesos
declare -a running_pids=()
declare -a running_files=()

# Función para lanzar un trabajo
launch_job() {
    local mrxs_file="$1"
    local job_number="$2"
    
    local filename=$(basename "$mrxs_file")
    echo "Lanzando trabajo #$job_number para $filename"
    
    # Ejecutar srun en segundo plano
    srun --gres=gpu:1 --mem=25G --exclude=gpic10 --cpus-per-task=8 --time=02:00:00 \
         python parcher_mrxs.py --input_dir "$mrxs_file" --patch_size 2048 --save_geojson --save_composite --skip_background --overlap 256 &
    
    local pid=$!
    running_pids+=($pid)
    running_files+=("$filename")
    
    echo "Trabajo #$job_number lanzado con PID $pid"
}

# Función para limpiar procesos terminados
cleanup_finished_jobs() {
    local new_pids=()
    local new_files=()
    
    for i in "${!running_pids[@]}"; do
        local pid=${running_pids[i]}
        local filename=${running_files[i]}
        
        # Verificar si el proceso sigue corriendo
        if kill -0 "$pid" 2>/dev/null; then
            # El proceso sigue corriendo
            new_pids+=("$pid")
            new_files+=("$filename")
        else
            # El proceso ha terminado
            echo "Trabajo completado: $filename (PID $pid)"
        fi
    done
    
    running_pids=("${new_pids[@]}")
    running_files=("${new_files[@]}")
}

# Contador para numeración de trabajos
job_counter=1

# Índice para el array de archivos
file_index=0

# Lanzar los primeros trabajos hasta el máximo permitido
while [ $file_index -lt ${#mrxs_array[@]} ] && [ ${#running_pids[@]} -lt $max_parallel ]; do
    launch_job "${mrxs_array[$file_index]}" $job_counter
    ((file_index++))
    ((job_counter++))
done

# Procesar los archivos restantes
while [ $file_index -lt ${#mrxs_array[@]} ]; do
    # Limpiar trabajos terminados
    cleanup_finished_jobs
    
    # Lanzar nuevos trabajos si hay espacio disponible
    while [ $file_index -lt ${#mrxs_array[@]} ] && [ ${#running_pids[@]} -lt $max_parallel ]; do
        launch_job "${mrxs_array[$file_index]}" $job_counter
        ((file_index++))
        ((job_counter++))
    done
    
    # Esperar un poco antes de verificar de nuevo
    sleep 400
    
    # Mostrar estado actual
    echo "Procesos activos: ${#running_pids[@]}, Archivos restantes: $((${#mrxs_array[@]} - file_index))"
done

# Esperar a que terminen todos los trabajos restantes
echo "Esperando a que terminen los trabajos restantes..."
while [ ${#running_pids[@]} -gt 0 ]; do
    cleanup_finished_jobs
    if [ ${#running_pids[@]} -gt 0 ]; then
        echo "Esperando a ${#running_pids[@]} trabajos restantes..."
        sleep 400
    fi
done

echo "Todos los trabajos han finalizado. Se procesaron un total de $((job_counter-1)) archivos."