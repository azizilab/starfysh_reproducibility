# spatial transcriptome analysis

## Problem setting: 

Spatial Transcriptomics (ST / Visium) data captures gene expressions as well as their locations. However, with limited spatial resolution, each spot usually covers more than 1 cells. To infer potential cellular interactions, we need to infer deconvoluted components specific to each cell-type from the spots to infer functional modules describing cellular states. 



## Models:
- Semi-supervised Autoencoder
Use spatial transcriptomics expression data & annotated signature gene sets as input; perform deconvolution and reconstructure features from the bottle neck neurons; we hope these could capture gene sets representing specific functional modules.


## Directories
```
.
├── archive:        Exploratory analysis notebooks & pipeline scripts
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample notebook & tutorial (to be updated)
├── run_PoE:        Pipeline notebooks to generate pre/post-processing & analysis figures
├── semiVAE_all:    Combined model ( i). expression-based deconvolution; ii). expression + image (PoE) deconvolution
├── VAE:            Previous expression-based deconvolution (to be cleaned)
```


## Installation
```bash
# install
python setup.py install --user

# uninstall
pip uninstall bcvae

# re-install
./reinstall.sh
```
