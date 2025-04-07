import openslide
import os
import glob

slide_path = "/mnt/work/datasets/BKidney/CROC/"

# Crear directorio de salida principal
os.makedirs(slide_path, exist_ok=True)

# Buscar todas las imágenes MRXS en el directorio de entrada
mrxs_files = glob.glob(os.path.join(slide_path, "*.mrxs"))
for slide_path in mrxs_files:
  slide = openslide.OpenSlide(slide_path)

  print(f"Niveles disponibles: {slide.level_count}")
  print(f"Dimensiones en cada nivel: {slide.level_dimensions}")
  print(f"📌 La imagen tiene {slide.level_count} niveles disponibles")
  for i, dim in enumerate(slide.level_dimensions):
      print(f"🔹 Nivel {i}: {dim[0]}x{dim[1]} píxeles")

  slide.close()
