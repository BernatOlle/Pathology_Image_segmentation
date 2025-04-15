from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
import os
import mmengine
from glob import glob

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
            img_suffix='_img.jpg',  # Sufijo específico para imágenes
            seg_map_suffix='_mask.jpg',  # Sufijo específico para máscaras
            reduce_zero_label=True,
            **kwargs
        )
        
    def load_data_list(self):
        """Load data list with automatic case directory discovery."""
        data_list = []
        img_root = os.path.join(self.data_root, self.data_prefix['img_path'])
        ann_root = os.path.join(self.data_root, self.data_prefix['seg_map_path'])
        
       
        
        # Encontrar todos los directorios de casos (56Nx, DN, etc.)
        case_dirs = [d for d in os.listdir(img_root) 
                    if os.path.isdir(os.path.join(img_root, d)) and not d.startswith('.')]
        
        if not case_dirs:
            raise RuntimeError(f"No case directories found in {img_root}")
            
        
        
        for case_dir in case_dirs:
            case_img_path = os.path.join(img_root, case_dir)
            case_ann_path = os.path.join(ann_root, case_dir)
            
            # Buscar subdirectorios (como 12-299)
            for subcase in os.listdir(case_img_path):
                
                
                subcase_img_path = os.path.join(case_img_path, subcase, 'img')
                subcase_ann_path = os.path.join(case_ann_path, subcase, 'mask')
                
                if not os.path.exists(subcase_img_path):
                    continue
                    
                # Buscar imágenes con el sufijo _img.jpg
                img_files = glob(os.path.join(subcase_img_path, f'*{self.img_suffix}'))
                
                for img_file in img_files:
                    # Construir nombre de la máscara (reemplazar _img.jpg por _mask.png)
                    base_name = os.path.basename(img_file).replace(self.img_suffix, '')
                    mask_file = os.path.join(subcase_ann_path, f"{base_name}{self.seg_map_suffix}")
                    
                    if not os.path.exists(mask_file):
                        print(f"Warning: Mask not found for {img_file}")
                        continue
                        
                    data_list.append({
                        'img_path': img_file,
                        'seg_map_path': mask_file,
                        'label_map': None,
                        'reduce_zero_label': self.reduce_zero_label,
                        'seg_fields': []
                    })
        
        if not data_list:
            raise RuntimeError(
                f"No valid image-mask pairs found in:\n"
                f"Image root: {img_root}\n"
                f"Mask root: {ann_root}\n"
                f"Image suffix: {self.img_suffix}\n"
                f"Mask suffix: {self.seg_map_suffix}"
            )
            
        
        return data_list