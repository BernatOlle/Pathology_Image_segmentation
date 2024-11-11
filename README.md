# Glomerulus Segmentation Pipeline

This repository includes the pipeline and pre-trained models for **patch-level** and **WSI-level** glomerulus segmentation.  
Models were trained on the [KPIs](https://sites.google.com/view/kpis2024/) and [mice glomeruli](https://datadryad.org/stash/dataset/doi:10.5061/dryad.fqz612jpc) datasets

## 🌟 Highlights 🌟
- #### 🎉 This pipeline won 1st place in <ins>both tracks</ins> of the Kidney Pathology Image segmentation ([KPIs](https://sites.google.com/view/kpis2024/)) challenge - MICCAI 2024 🥇. See the code and solution [here](KPIs2024/Solution.md)
- #### Benchmarking on the KPIs and Mice glomeruli datasets. Pre-trained segmentation models are available publicly [here](docs/model_zoo.md)

## Installation
We use mmsegmentation package, please follow the [installation guideline](docs/get_started.md) to install the inference code.

## Model Zoo
All pre-trained models are available in [model zoo](docs/model_zoo.md)

## Datasets
See [datasets](docs/datasets_howto.md) for download and pre-process the KPIs and mice glomeruli datasets

## Inference
### Inference on a single input
```python
import cv2
from mmseg.apis import init_model, inference_model
from mmengine.registry import init_default_scope

# init the default transform to mmseg
init_default_scope('mmseg')

# define test_pipeline
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='PackSegInputs'),
]

# example: load SegFormer model
config_path = 'segformer_mit-b5_kpis_isbi_768.py'
ckpt_path = 'segformer_mit_b5_kpis_768_best_mDice.pth'
model = init_model(config_path, ckpt_path)

# assign test_pipeline
model.cfg.test_pipeline = test_pipeline

# inference
img_data = cv2.imread('/path/to/your/image', -1)
pred_res = inference_model(model, img_data)

# get the predicted mask
raw_logits = pred_res.seg_logits.data
_, pred_mask = raw_logits.max(axis=0, keepdims=True)
pred_mask = pred_mask.cpu().numpy()[0]
```

### Inference & calculate Dice on patch-level data
Accept single image input or a directory of image patches (all patches must be from the same WSI).  
Images and their ground-truth masks must be inside the `/img/` and `/mask/` directories.  
For example, inference and apply stitching strategy
```bash
python inference_patch_level.py \
--input KPIs24_Testing_Data/Task1_patch_level/test/DN/11-363 \
--config segformer_mit-b5_kpis_768/segformer_mit-b5_kpis_isbi_768.py  \
--ckpt segformer_mit-b5_kpis_768/segformer_mit_b5_kpis_768_best_mDice.pth \
--img_size 2048 --stitch
```

### Inference & calculate Dice on WSI-level data
Accept a single WSI input. 
WSI and its ground-truth mask must be available.  

Inference on KPIs WSI
```bash
python inference_wsi_level_kpis.py \
--input KPIs24_Testing_Data/Task2_WSI_level/NEP25/18-579_wsi.tiff \
--config segformer_mit-b5_kpis_768/segformer_mit-b5_kpis_isbi_768.py \
--ckpt segformer_mit-b5_kpis_768/segformer_mit_b5_kpis_768_best_mDice.pth \
--patch_size 2048 --stride 1024
```

Inference on Mice glomeruli WSI
```bash
python inference_wsi_level_mice_glomeruli.py \
--input Orbit_Glomeruli/test/4730025.tiff \
--config mask2former_swin-b_orbit_768/mask2former_swin-b_orbit_isbi_768.py \
--ckpt mask2former_swin-b_orbit_768/best_mDice_iter_33520.pth \
--patch_size 1024 --stride 512
```

### TO-DO
- [ ] Add notebook examples on how to process data and inference

## Citation
```
@article{cap24wsiglomerulus,
  title   = {An Effective Pipeline for Whole-Slide Image Glomerulus Segmentation},
  author  = {Quan Huu Cap},
  journal = {arXiv preprint arXiv:2411.04782},
  year    = {2024}
}
```