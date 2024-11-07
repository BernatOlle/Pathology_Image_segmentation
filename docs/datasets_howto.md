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
This dataset comprises 88 WSIs with edge sizes ranging from 14,000 to 65,000 pixels. 
The histopathological data come from mice and rats.  
### How to download
**Image data**: https://datadryad.org/stash/dataset/doi:10.5061/dryad.fqz612jpc  
**Annotation data**: https://drive.google.com/file/d/1rXacSs_7e3cnbHpHttTn539U1f_MgSEq/view?usp=sharing  
Note that the mask data is extracted from the original annotation data in SQLite database format.  
We manually extract patches of size $1024\times1024$ and make sure each patch image contains at least one glomerulus annotation. 
For WSI-level evaluation data, we manually checked and excluded the WSIs that are possibly incomplete in labeling, resulting in 27 WSI-level ground-truth masks.  
Please check out [notebooks](notebooks) or [scripts](scripts) to see how to pre-process this dataset.