# Get started: Install `mmseg`
## Prerequisites
- Python 3.9 (e.g., conda or virtual envs)
- CUDA 11.6
- cuDNN 8.8
- nccl 2.16

**Step 1.** Create a conda environment and activate it

```shell
conda create --name mmseg_wsi python=3.9 -y
conda activate mmseg_wsi
```

**Step 2.** Install PyTorch

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

**Step 3.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim)

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
```

**Step 3.** Install other necessary packages

```shell
pip install ftfy regex tqdm
pip install "albumentations==1.4.6" --no-binary qudida,albumentations
pip install "numpy<2"
```

**Step 5.** Install MMSegmentation, MMDet, MMPreTrain

```shell
pip install "mmsegmentation==1.2.2"
pip install "mmdet>=3.0.0rc4"
pip install "mmpretrain>=1.0.0rc7"
```