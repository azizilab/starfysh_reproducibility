<img src=logo.png width="500" />

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
  - Spot-wise deconvolution matrix (`q(c)`)
  - Low-dimensional manifold representation (`q(z)`)
  - Clusterings (single-sample) / Hubs (multiple-sample integration) given the deconvolution results
  - Co-localization networks across cell types and Spatial R-L interactions
  - Imputated count matrix (`p(x)`)

## Directories
```
.
├── data:           Spatial Transcritomics & synthetic simulation datasets
├── notebooks:      Sample notebook & tutorial
├── simulation:     Synthetic simulation from scRNA-seq for benchmark
├── starfysh:       Starfysh core model
```

## Installation
```bash
# install
python setup.py install --user

# uninstall
pip uninstall starfysh
```

## Quickstart
```python
import numpy as np
import pandas as pd
import torch
from starfysh import (
    AA,
    dataloader,  
    starfysh,
    utils,
    plot_utils,
    post_analysis
)

# (1) Loading dataset & signature gene sets
data_path = 'data/' # specify data directory
sig_path = 'signature/signatures.csv' # specify signature directory
sample_id = 'CID44971_TNBC'

# --- (a) ST matrix ---
adata, adata_norm = utils.load_adata(
    data_path,
    sample_id,
    n_genes=2000
)

# --- (b) paired H&E image + spots info ---
hist_img, map_info = utils.preprocess_img(
    data_path,
    sample_id,
    adata.obs.index,
    hchannal=False
)

# --- (c) signature gene sets ---
gene_sig = utils.filter_gene_sig(
    pd.read_csv(sig_path),
    adata.to_df()
)

# (2) Starfysh deconvolution

# --- (a) Preparing arguments for model training
args = utils.VisiumArguments(
    adata,
    adata_norm,
    gene_sig,
    map_info,
    n_anchors=60, # number of anchor spots per cell-type
    window_size=5  # library size smoothing radius
)

adata, adata_noprm = args.get_adata()

# --- (b) Model training ---
n_restarts = 3
epochs = 100
patience = 10
device = torch.device('cpu')

model, loss = utils.run_starfysh(
    args,
    n_restarts,
    epochs=epochs,
    patience=patience
)

# (3). Parse deconvolution outputs
inferences, generatives, px = starfysh.model_eval(
    model,
    adata,
    args.sig_mean,
    device,
    args.log_lib,
)


```
