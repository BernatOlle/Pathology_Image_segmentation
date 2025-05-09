import os
import glob
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import pandas as pd
from datetime import datetime
import json
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Import from mmseg and mmengine
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluar el rendimiento de un modelo de segmentación de glomérulos'
    )
    parser.add_argument('--test_dir', type=str, required=True, help='Directorio con los datos de prueba')
    parser.add_argument('--config_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mask2Former/mask2former_swin-b_kpis_isbi_768.py', 
                        help='Ruta del archivo de configuración del modelo')
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/usuaris/imatge/bernat.olle/Pathology_Image_segmentation/mmsegmentation/mask2former_swin-b_kpis_768/best_mDice_iter_6000.pth', 
                        help='Ruta del archivo de checkpoint del modelo')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                        help='Directorio donde guardar los resultados de la evaluación')
    parser.add_argument('--visualize', action='store_true', 
                        help='Guardar visualizaciones de las predicciones vs ground truth')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Umbral para binarizar las máscaras de predicción')
    parser.add_argument('--glomeruli_class', type=int, default=1, 
                        help='Índice de la clase de glomérulos en la salida del modelo')
    return parser.parse_args()

def initialize_model(config_path, ckpt_path, device='cuda:0'):
    """
    Inicializa el modelo Mask2Former para segmentación de glomérulos
    usando mmseg y mmengine
    """
    # Inicializar el ámbito predeterminado para mmseg
    init_default_scope('mmseg')
    
    # Cargar el modelo
    model = init_model(config_path, ckpt_path, device=device)
    
    # Definir el pipeline de prueba
    test_pipeline = [
        dict(type='LoadImageFromNDArray'),
        dict(type='PackSegInputs'),
    ]
    
    # Asignar el pipeline al modelo
    model.cfg.test_pipeline = test_pipeline
    
    return model

def get_glomeruli_mask(result, target_height, target_width, glomeruli_class=1):
    """
    Convierte el resultado de la inferencia en una máscara binaria de glomérulos
    """
    # Obtener los logits de segmentación
    raw_logits = result.seg_logits.data
    
    # Obtener la clase con mayor probabilidad
    _, pred_mask = raw_logits.max(axis=0, keepdims=True)
    pred_mask = pred_mask.cpu().numpy()[0]
    
    # Convertir a imagen binaria para la clase de glomérulos especificada
    binary_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    binary_mask[pred_mask == glomeruli_class] = 255
    
    # Redimensionar al tamaño objetivo si es necesario
    if binary_mask.shape[0] != target_height or binary_mask.shape[1] != target_width:
        binary_mask = cv2.resize(binary_mask, (target_width, target_height), 
                                interpolation=cv2.INTER_NEAREST)
    
    return binary_mask

def calculate_metrics(pred_mask, gt_mask):
    """
    Calcula métricas de evaluación entre la máscara predicha y el ground truth
    """
    # Normalizar las máscaras a valores binarios (0-1)
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    
    # Aplanar las máscaras para calcular métricas basadas en píxeles
    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()
    
    # Calcular métricas
    metrics = {}
    
    # Dice coefficient (F1-score)
    dice = f1_score(gt_flat, pred_flat, zero_division=1)
    metrics['dice'] = dice
    
    # IoU (Jaccard index)
    iou = jaccard_score(gt_flat, pred_flat, zero_division=1)
    metrics['iou'] = iou
    
    # Precision
    precision = precision_score(gt_flat, pred_flat, zero_division=1)
    metrics['precision'] = precision
    
    # Recall (Sensitivity)
    recall = recall_score(gt_flat, pred_flat, zero_division=1)
    metrics['recall'] = recall
    
    # Specificity
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    metrics['specificity'] = specificity
    
    # Accuracy
    accuracy = np.mean(pred_flat == gt_flat)
    metrics['accuracy'] = accuracy
    
    # Área de la predicción y ground truth (en píxeles)
    pred_area = np.sum(pred_bin)
    gt_area = np.sum(gt_bin)
    metrics['pred_area'] = pred_area
    metrics['gt_area'] = gt_area
    
    # Proporción de área (predicted/ground truth)
    area_ratio = pred_area / gt_area if gt_area > 0 else 0
    metrics['area_ratio'] = area_ratio
    
    # Hausdorff distance (distancia máxima entre contornos)
    try:
        # Encontrar contornos
        pred_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si hay contornos en ambas máscaras
        if pred_contours and gt_contours:
            # Calcular las distancias entre todos los puntos de los contornos
            min_hausdorff = float('inf')
            
            for pred_contour in pred_contours:
                for gt_contour in gt_contours:
                    # Convertir contornos a arrays de puntos
                    pred_points = pred_contour.reshape(-1, 2)
                    gt_points = gt_contour.reshape(-1, 2)
                    
                    # Calcular la distancia para cada par de puntos
                    current_hausdorff = 0
                    for pred_point in pred_points:
                        min_dist = float('inf')
                        for gt_point in gt_points:
                            dist = np.linalg.norm(pred_point - gt_point)
                            min_dist = min(min_dist, dist)
                        current_hausdorff = max(current_hausdorff, min_dist)
                    
                    min_hausdorff = min(min_hausdorff, current_hausdorff)
            
            metrics['hausdorff'] = min_hausdorff
        else:
            metrics['hausdorff'] = float('inf') if (gt_area > 0 or pred_area > 0) else 0
    except Exception as e:
        print(f"Error al calcular la distancia de Hausdorff: {e}")
        metrics['hausdorff'] = float('nan')
    
    return metrics

def visualize_results(img_path, pred_mask, gt_mask, output_path):
    """
    Crea una visualización comparativa de la predicción vs ground truth
    """
    # Cargar la imagen original
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preparar las máscaras para visualización
    pred_mask_vis = (pred_mask > 0).astype(np.uint8) * 255
    gt_mask_vis = (gt_mask > 0).astype(np.uint8) * 255
    
    # Crear overlay para predicción (en rojo)
    pred_overlay = img.copy()
    pred_overlay[pred_mask_vis > 0] = [255, 0, 0]  # Rojo para predicción
    
    # Crear overlay para ground truth (en verde)
    gt_overlay = img.copy()
    gt_overlay[gt_mask_vis > 0] = [0, 255, 0]  # Verde para ground truth
    
    # Crear overlay combinado
    combined_overlay = img.copy()
    # Verde para ground truth
    combined_overlay[gt_mask_vis > 0] = [0, 255, 0]
    # Rojo para falsos positivos (predicción pero no ground truth)
    combined_overlay[(pred_mask_vis > 0) & (gt_mask_vis == 0)] = [255, 0, 0]
    # Amarillo para verdaderos positivos (predicción y ground truth)
    combined_overlay[(pred_mask_vis > 0) & (gt_mask_vis > 0)] = [255, 255, 0]
    
    # Crear figura de 2x2
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Imagen original
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Imagen Original')
    axs[0, 0].axis('off')
    
    # Predicción
    axs[0, 1].imshow(pred_overlay)
    axs[0, 1].set_title('Predicción')
    axs[0, 1].axis('off')
    
    # Ground Truth
    axs[1, 0].imshow(gt_overlay)
    axs[1, 0].set_title('Ground Truth')
    axs[1, 0].axis('off')
    
    # Overlay combinado
    axs[1, 1].imshow(combined_overlay)
    axs[1, 1].set_title('Comparación (Verde: GT, Rojo: FP, Amarillo: TP)')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def evaluate_dataset(test_dir, model, output_dir, visualize=False, threshold=0.5, glomeruli_class=1):
    """
    Evalúa el modelo en todo el conjunto de datos de prueba
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Directorio para visualizaciones
    vis_dir = os.path.join(output_dir, 'visualizations')
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Lista para almacenar resultados
    all_results = []
    
    # Obtener todas las carpetas Sxx
    slide_folders = [f for f in os.listdir(test_dir) if f.startswith('S') and os.path.isdir(os.path.join(test_dir, f))]
    
    print(f"Encontradas {len(slide_folders)} carpetas de slides para evaluar")
    
    for slide_folder in sorted(slide_folders):
        slide_path = os.path.join(test_dir, slide_folder)
        
        # Verificar que existan las carpetas img y mask
        img_dir = os.path.join(slide_path, 'img')
        mask_dir = os.path.join(slide_path, 'mask')
        
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"Advertencia: No se encontraron directorios img/mask en {slide_path}")
            continue
        
        # Obtener todas las imágenes
        img_files = glob.glob(os.path.join(img_dir, '*img.png'))
        
        print(f"Procesando {slide_folder}: {len(img_files)} imágenes")
        
        slide_results = []
        
        for img_path in tqdm(img_files, desc=f"Evaluando {slide_folder}"):
            # Construir la ruta a la máscara correspondiente
            img_filename = os.path.basename(img_path)
            mask_filename = img_filename.replace('img.png', 'mask.png')
            mask_path = os.path.join(mask_dir, mask_filename)
            
            # Verificar que la máscara existe
            if not os.path.exists(mask_path):
                print(f"Advertencia: No se encontró máscara para {img_path}")
                continue
            
            # Cargar imagen y máscara
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Asegurarse de que la máscara sea binaria
            gt_mask = (gt_mask > 0).astype(np.uint8) * 255
            
            # Inferencia del modelo
            try:
                result = inference_model(model, img)
                
                # Obtener la máscara de glomérulos
                pred_mask = get_glomeruli_mask(result, gt_mask.shape[0], gt_mask.shape[1], glomeruli_class)
                
                # Calcular métricas
                metrics = calculate_metrics(pred_mask, gt_mask)
                
                # Añadir información de la imagen y slide
                result_entry = {
                    'slide_id': slide_folder,
                    'image_path': img_path,
                    'mask_path': mask_path,
                    **metrics
                }
                
                slide_results.append(result_entry)
                
                # Visualizar resultados si se solicita
                if visualize:
                    vis_filename = f"{slide_folder}_{os.path.basename(img_path).replace('.png', '_eval.png')}"
                    vis_path = os.path.join(vis_dir, vis_filename)
                    visualize_results(img_path, pred_mask, gt_mask, vis_path)
                
            except Exception as e:
                print(f"Error al procesar {img_path}: {e}")
        
        # Calcular métricas promedio para el slide actual
        if slide_results:
            slide_avg = {metric: np.mean([r[metric] for r in slide_results if metric in r]) 
                         for metric in ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy']}
            
            print(f"Resultados para {slide_folder}:")
            print(f"  Dice: {slide_avg['dice']:.4f}")
            print(f"  IoU: {slide_avg['iou']:.4f}")
            print(f"  Precision: {slide_avg['precision']:.4f}")
            print(f"  Recall: {slide_avg['recall']:.4f}")
            print(f"  Specificity: {slide_avg['specificity']:.4f}")
            print(f"  Accuracy: {slide_avg['accuracy']:.4f}")
            
            # Añadir todos los resultados del slide a la lista general
            all_results.extend(slide_results)
    
    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Guardar resultados en CSV
    csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Calcular y guardar métricas promedio totales
    metrics_cols = ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy']
    
    # Métricas globales
    global_metrics = {metric: results_df[metric].mean() for metric in metrics_cols}
    
    # Métricas por slide
    slide_metrics = results_df.groupby('slide_id')[metrics_cols].mean().reset_index()
    
    # Guardar métricas globales
    with open(os.path.join(output_dir, 'global_metrics.json'), 'w') as f:
        json.dump(global_metrics, f, indent=4)
    
    # Guardar métricas por slide
    slide_metrics.to_csv(os.path.join(output_dir, 'slide_metrics.csv'), index=False)
    
    # Generar gráficos de métricas por slide
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics_cols):
        plt.subplot(2, 3, i+1)
        plt.bar(slide_metrics['slide_id'], slide_metrics[metric])
        plt.title(f'{metric.capitalize()} por Slide')
        plt.xlabel('Slide ID')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'metrics_by_slide.png'), dpi=150)
    plt.close()
    
    # Generar un informe de evaluación en HTML
    generate_html_report(output_dir, global_metrics, slide_metrics)
    
    print(f"\nEvaluación completa. Resultados guardados en {output_dir}")
    
    # Imprimir métricas globales
    print("\nMétricas globales:")
    for metric, value in global_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    return global_metrics

def generate_html_report(output_dir, global_metrics, slide_metrics):
    """
    Genera un informe HTML con los resultados de la evaluación
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Evaluación - Segmentación de Glomérulos</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics-container {{ display: flex; flex-wrap: wrap; }}
            .metric-box {{ 
                flex: 1; 
                min-width: 200px; 
                margin: 10px; 
                padding: 15px; 
                background-color: #f0f0f0; 
                border-radius: 5px;
                text-align: center;
            }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .metric-name {{ font-size: 14px; color: #666; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Informe de Evaluación - Segmentación de Glomérulos</h1>
            <p>Fecha de generación: {now}</p>
            
            <h2>Métricas Globales</h2>
            <div class="metrics-container">
    """
    
    # Añadir métricas globales
    for metric, value in global_metrics.items():
        html_content += f"""
                <div class="metric-box">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-name">{metric.capitalize()}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Métricas por Slide</h2>
            <table>
                <thead>
                    <tr>
                        <th>Slide ID</th>
                        <th>Dice</th>
                        <th>IoU</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>Specificity</th>
                        <th>Accuracy</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Añadir métricas por slide
    for _, row in slide_metrics.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row['slide_id']}</td>
                        <td>{row['dice']:.4f}</td>
                        <td>{row['iou']:.4f}</td>
                        <td>{row['precision']:.4f}</td>
                        <td>{row['recall']:.4f}</td>
                        <td>{row['specificity']:.4f}</td>
                        <td>{row['accuracy']:.4f}</td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
            
            <h2>Gráficos</h2>
            <img src="metrics_by_slide.png" alt="Métricas por Slide">
            
            <h2>Visualizaciones</h2>
            <p>Las visualizaciones detalladas están disponibles en la carpeta 'visualizations' del directorio de salida.</p>
        </div>
    </body>
    </html>
    """
    
    # Guardar el informe HTML
    with open(os.path.join(output_dir, 'evaluation_report.html'), 'w') as f:
        f.write(html_content)

def main():
    args = parse_args()
    
    # Validar argumentos
    test_dir = Path(args.test_dir)
    if not test_dir.exists() or not test_dir.is_dir():
        print(f"Error: El directorio de prueba {args.test_dir} no existe")
        return
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar el modelo
    print(f"Inicializando modelo desde {args.config_path}")
    model = initialize_model(args.config_path, args.ckpt_path)
    print("Modelo cargado correctamente")
    
    # Evaluar el dataset
    evaluate_dataset(
        test_dir=args.test_dir,
        model=model,
        output_dir=args.output_dir,
        visualize=args.visualize,
        threshold=args.threshold,
        glomeruli_class=args.glomeruli_class
    )
    
    print(f"\n✅ Evaluación completada. Resultados guardados en: {args.output_dir}")

if __name__ == "__main__":
    main()