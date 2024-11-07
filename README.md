# Glomerulus Segmentation Pipeline - 1st Place in Both Tracks of the Kidney Pathology Image Segmentation ([KPIs](https://sites.google.com/view/kpis2024/)) Challenge 2024

This repository includes the pipeline and pre-trained models for patch-level and WSI-level glomerulus segmentation.  
All models were trained on the [KPIs](https://sites.google.com/view/kpis2024/) and [mice glomeruli](https://datadryad.org/stash/dataset/doi:10.5061/dryad.fqz612jpc) datasets

## 🌟 Highlights
This pipeline won 1st place in both tracks of the Kidney Pathology Image segmentation ([KPIs](https://sites.google.com/view/kpis2024/)) challenge - MICCAI 2024. See the code and solution [here](KPIs2024/Solution.md)

## Installation
We use mmsegmentation package, please follow the [installation guideline](docs/get_started.md) to install the inference code.

## Model Zoo
Please see all pre-trained models in the [model zoo](docs/model_zoo.md)

## Datasets
See [datasets](docs/datasets_howto.md) for download and pre-process the KPIs and mice glomeruli datasets

## Inference
The following shows an example of loading and inferencing a model on a single input
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
### Notebooks and more code will be released soon ...