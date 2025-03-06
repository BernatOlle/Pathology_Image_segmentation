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