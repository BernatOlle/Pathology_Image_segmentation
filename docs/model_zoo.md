# Pre-trained segmentation models on the KPIs and mice glomeruli datasets
Each pre-trained model comes with a model weight and a config file. 
To use the model, please see [notebooks](../notebooks/) and [scripts](../scripts/)

### KPIs
| Method      | Backbone   | Patch-level | WSI-level | Download |
|-------------|------------|-------------|-----------|----------|
| U-Net       | U-Net      | 88.54       | 86.62     | [link](https://drive.google.com/drive/folders/1SEdeM66uoUGXB3MJHtkP7pIYNaGzzvV7?usp=drive_link)      |
| DeepLabV3+  | ResNet-50  | 90.85       | 91.58     | [link](https://drive.google.com/drive/folders/1pCiQJpzSnN598crCWFgaml5c95SqL5AW?usp=drive_link)      |
| SegFormer   | MIT-B5     | 94.35       | 94.51     | [link](https://drive.google.com/drive/folders/1sl6EVmi-RWJVTtVP3iyfTxx66oESZQ0l?usp=drive_link)      |
| UPerNet     | ConvNeXt-B | 94.29       | 94.23     | [link](https://drive.google.com/drive/folders/1v5ChDDsRluTtv8LN4UCCWxv4CRQFCs9b?usp=drive_link)      |
| Mask2Former | Swin-B     | **94.42**       | **94.64**     | [link](https://drive.google.com/drive/folders/1ORYOhx-dqZ5CTeDLvwTSgLsueW4phBNJ?usp=drive_link)      |

### Mice glomeruli
| Method      | Backbone   | Patch-level | WSI-level | Download |
|-------------|------------|-------------|-----------|----------|
| U-Net       | U-Net      | 89.58       | 83.66     | [link](https://drive.google.com/drive/folders/1BzcdrIOev1EFr0EWcqbpyg1p9J7SzWCx?usp=drive_link)     |
| DeepLabV3+  | ResNet-50  | 90.49       | 80.32     | [link](https://drive.google.com/drive/folders/13cW0LHb42h3pOFh-rx7C6piXNU8qkP4P?usp=drive_link)     |
| SegFormer   | MIT-B5     | 91.84       | 79.69     | [link](https://drive.google.com/drive/folders/1BuYUH-nU6w3IpGS05UeByk8aHPBicCjF?usp=drive_link)     |
| UPerNet     | ConvNeXt-B | **92.10**   | **86.67** | [link](https://drive.google.com/drive/folders/1jS3TZBRzFH8KxqiX7KS6342UHP8bIhHP?usp=drive_link)     |
| Mask2Former | Swin-B     | 91.98       | 86.44     | [link](https://drive.google.com/drive/folders/1O9D0vsUKIsQzby4bNRLq22bxJ_khLA1q?usp=drive_link)     |