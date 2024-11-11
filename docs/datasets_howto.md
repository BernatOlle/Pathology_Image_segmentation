# How to process the KPIs and mice glomeruli datasets

## KPIs dataset
This dataset includes 50 high-resolution WSIs of whole mouse kidneys and is released along with the [KPIs challenge 2024](https://sites.google.com/view/kpis2024/).
This is a well-prepared dataset and no preprocessing is needed. There are already split training/validataion/testing sets. Each path image has the size of $2048\times2048$. See the KPIs challenge's website for more details.

### How to download
To to https://www.synapse.org/Synapse:syn54077668 and register an account.  
Once you have your account, install the [synapse](https://pypi.org/project/synapse/) package to download via command line.
```shell
pip install synapse
```
To download, use the command
```shell
synapse get -r syn54077668
```

## Mice glomeruli dataset
This dataset comprises 88 WSIs and is released along with the [Orbit Image Analysis tool](https://www.orbit.bio/). 
The histopathological data come from mice and rats.  

### How to download
**Image data**: https://datadryad.org/stash/dataset/doi:10.5061/dryad.fqz612jpc  
**Annotation data**: https://drive.google.com/file/d/1rXacSs_7e3cnbHpHttTn539U1f_MgSEq/view?usp=sharing  

Note that the mask data is extracted from the original annotation data in SQLite database format.  
We manually extract patches of size $1024\times1024$ and make sure each patch image contains at least one glomerulus annotation. 
For WSI-level evaluation data, we manually checked and excluded the WSIs that are possibly incomplete in labeling, resulting in 27 WSI-level ground-truth masks.  

### Preparing the data
**Step 1**: Download the above image and annotation data and extract them somewhere. Let's call them the `mice_glomeruli_root` and  `seg_ann_data`.  

**Step 2**: Run the [preprocess script](../scripts/preprocess_mice_glomeruli.py) to extract patch-level data and WSI-level mask data. 
```bash
python preprocess_mice_glomeruli.py --data_root mice_glomeruli_root --ann_root seg_ann_data --crop_size 1024
```

After processing, the data structure looks like below:
```bash
# WSIs data
mice_glomeruli_root
└── test
└── train

# WSIs ground-truth masks
mice_glomeruli_root
└── test_mask
└── train_mask

# patch-level data
mice_glomeruli_root
└── extracted_data
    └── Testing_data_patch
        └── WSI_group
            └── WSI_ID
                └── img
                └── mask
            ...
        ...
    └── Training_data_patch
        └── WSI_group
            └── WSI_ID
                └── img
                └── mask
            ...
        ...
```

Please take a look at the [notebooks](../notebooks/) for more detailed examples.