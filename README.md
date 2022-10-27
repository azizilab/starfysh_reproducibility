<img src=https://github.com/azizilab/Starfysh/blob/main/logo.png width="500" />

# Spatial Transcriptomic Analysis using Reference-Free auxiliarY deep generative modeling and Shared Histology


## Problem setting: 

Spatial Transcriptomics (ST / Visium) data captures gene expressions as well as their locations. However, with limited spatial resolution, each spot usually covers more than 1 cells. To infer potential cellular interactions, we need to infer deconvoluted components specific to each cell-type from the spots to infer functional modules describing cellular states. 



## Models:
- Semi-supervised learning with Auxiliary Variational Autoencoder (AVAE) for cell-type deconvolution
- Archetypal analysis for unsupervised cell-type discovery (novel cell types) & marker gene refinement (existing annotated cell types)
- Product-of-Experts (PoE) for H&E image integration

- Input:
  - Spatial Transcriptomics count matrix
  - Annotated signature gene sets
  - (Optional): paired H&E image
  
- Output:
  - Spot-wise deconvolution matrix ($q_{\phi}(c)$)
  - Low-dimensional manifold representation ($q_{\phi}(z)$)
  - Clusterings (single-sample) / Hubs (multiple-sample integration) given the deconvolution results
  - Co-localization networks across cell types and Spatial R-L interactions
  - Imputated count matrix ($p(x \mid z)$)

## Directories
```
.
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample notebook & tutorial
├── simulation:     Synthetic simulation from scRNA-seq for benchmark
├── starfysh:       Starfysh core model
```

## Installation
(To be updated: currently only contain expression-based deconvolution model)
```bash
# install
python setup.py install --user

# uninstall
pip uninstall bcvae

# re-install
./reinstall.sh
```
