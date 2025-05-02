from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
import os
import mmengine
from glob import glob
from mmengine.logging import print_log

@DATASETS.register_module()
class KPIsDataset(BaseSegDataset):
    """Dataset class for Kidney Pathology Image Segmentation (KPIS)."""
    
    METAINFO = {
        'classes': ('background', 'kidney_structure'),
        'palette': [[0, 0, 0], [255, 255, 255]]
    }
    
    def __init__(self, **kwargs):
        kwargs.pop('reduce_zero_label', None)
        super().__init__(
            img_suffix='_img.png',  # Sufijo específico para imágenes
            seg_map_suffix='_mask.png',  # Sufijo específico para máscaras
            reduce_zero_label=True,
            **kwargs
        )
        
    # Ya no necesitamos el método find_folder para esta estructura simplificada
    
    def load_data_list(self):
        """Load data list for simplified directory structure where each case has img and mask folders."""
        data_list = []
        img_root = os.path.join(self.data_root, self.data_prefix['img_path'])
        ann_root = os.path.join(self.data_root, self.data_prefix['seg_map_path'])
        
        # Find all case directories (S24, S25, etc.)
        case_dirs = [d for d in os.listdir(img_root) 
                    if os.path.isdir(os.path.join(img_root, d)) and not d.startswith('.')]
        
        if not case_dirs:
            raise RuntimeError(f"No case directories found in {img_root}")
        
        print_log(f"Found {len(case_dirs)} case directories in {img_root}: {case_dirs}", "current")
            
        for case_dir in case_dirs:
            # Construir la ruta del directorio de imágenes y máscaras directamente
            case_img_path = os.path.join(img_root, case_dir, 'img')
            case_mask_path = os.path.join(ann_root, case_dir, 'mask')
            
            # Verificar si existen los directorios
            if not os.path.exists(case_img_path):
                print_log(f"No 'img' folder found in {os.path.join(img_root, case_dir)} - skipping case", "current")
                continue
                
            if not os.path.exists(case_mask_path):
                print_log(f"No 'mask' folder found in {os.path.join(ann_root, case_dir)} - skipping case", "current")
                continue
            
            # Buscar imágenes con el sufijo _img.png
            print_log(f"Looking for images in: {case_img_path} with suffix {self.img_suffix}", "current")
            img_files = glob(os.path.join(case_img_path, f'*{self.img_suffix}'))
            print_log(f"Found {len(img_files)} image files in {case_dir}/img", "current")
            
            for img_file in img_files:
                # Construir nombre de la máscara (reemplazar _img.png por _mask.png)
                base_name = os.path.basename(img_file).replace(self.img_suffix, '')
                mask_file = os.path.join(case_mask_path, f"{base_name}{self.seg_map_suffix}")
                
                if not os.path.exists(mask_file):
                    print_log(f"Warning: Mask not found for {img_file} - skipping", "current")
                    continue
                
                data_list.append({
                    'img_path': img_file,
                    'seg_map_path': mask_file,
                    'label_map': None,
                    'reduce_zero_label': self.reduce_zero_label,
                    'seg_fields': []
                })
        
        print_log(f"Total valid pairs found: {len(data_list)}", "current")
        
        if not data_list:
            raise RuntimeError(
                f"No valid image-mask pairs found in:\n"
                f"Image root: {img_root}\n"
                f"Mask root: {ann_root}\n"
                f"Image suffix: {self.img_suffix}\n"
                f"Mask suffix: {self.seg_map_suffix}"
            )
        
        return data_list