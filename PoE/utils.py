import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors

# preprocessing - remove Ribosomal & Mitochondrial genes
def exp_preprocess(adata_raw,
               min_perc=None,
               max_perc=None,
               n_top_genes=6000,
               mt_threshold=20):
    adata = adata_raw.copy()
    
    if min_perc and max_perc:
        assert 0 < min_perc < max_perc < 100,\
            "Invalid thresholds for cells: {0}, {1}".format(min_perc, max_perc)
        min_counts = np.percentile(adata.obs['total_counts'], min_perc)
        max_counts = np.percentile(adata.obs['total_counts'], min_perc)
        sc.pp.filter_cells(adata, min_counts=min_counts, max_counts=max_counts)
    
    # Remove cells with excessive MT expressions
    # Remove MT & RB genes
    print('Preprocessing1: delete the mt and rp')
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['rb'] = np.logical_or(
        adata.var_names.str.startswith('RPS'),
        adata.var_names.str.startswith('RPL')
    )
    
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < 100#mt_threshold
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])
    
    adata = adata[mask_cell, mask_gene] 

    
    # Preprocessing2: Normalize
    print('Preprocessing2: Normalize')
    sc.pp.normalize_total(adata, inplace=True) 

    # Preprocessing3: Logarithm
    print('Preprocessing3: Logarithm')
    sc.pp.log1p(adata)

    # Preprocessing4: Find the variable genes
    print('Preprocessing4: Find the variable genes')
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, inplace=True)
    
    return adata


#def plot_spatial_map(map_info, plot_variable):
import cv2
def img_preprocess(adata_image):
    
    lab= cv2.cvtColor(adata_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_image = np.dot(final[...,:3], rgb_weights)
    grayscale_image = (grayscale_image -grayscale_image.min())/(grayscale_image.max()-grayscale_image.min())
    grayscale_image = np.clip(grayscale_image, 0.2, 0.8)
    return grayscale_image

def get_mapinfo(adata_sample,sample_id):
    spot_diameter_fullres = adata_sample.uns['spatial'][sample_id]['scalefactors']['spot_diameter_fullres']
    tissue_hires_scalef = adata_sample.uns['spatial'][sample_id]['scalefactors']['tissue_hires_scalef']
    circle_radius = spot_diameter_fullres *tissue_hires_scalef * 0.5 
    image_spot = adata_sample.obsm['spatial'] *tissue_hires_scalef
    map_info = adata_sample.obs[['array_col','array_row']]
    map_info.loc[:,'imagecol'] = image_spot[:,0]
    map_info.loc[:,'imagerow'] = image_spot[:,1]
    
    return map_info


# deadling with gsva and siganture
def discretize_gsva(gsva_raw, thld=0.7):
    """
    Discretize GSVA scores for training
    
    Parameters
    ----------
    gsva_raw : pd.DataFrame
        raw GSVA matrix (shape: [S x C])
    
    """
    gsva_train = gsva_raw.apply(
        lambda x: x >= x.quantile(thld), 
        axis=0
    ).astype(np.uint8)
    
    return gsva_train


def find_spots(adata, gsva, n_nbrs=15):
    """
    Find spots representing `pure spots` & their nearest neighbors
    based on GSVA scores for each cell type
    """
    assert 'highly_variable' in adata.var_keys(), \
        "Please choose highly variable genes first"
    
    # Calculate distance with only highly variable genes
    embedding = adata[:, adata.var['highly_variable']].X.A
    pure_idxs = np.argmax(gsva.values, axis=0)
    pure_spots = gsva.idxmax(axis=0)
    
    pure_dict = {
        spot: ct 
        for (spot, ct) in zip(pure_spots, gsva.columns)
    }
    
    nbrs = NearestNeighbors(n_neighbors=n_nbrs+len(pure_spots)).fit(embedding)
    nn_graph = nbrs.kneighbors(embedding)[1] 
    
    perif_spots = []        
    perif_idxs = [nn_graph[idx].tolist() for idx in pure_idxs]
    for i, raw_idxs in enumerate(perif_idxs):
        idxs = [idx 
                for idx in raw_idxs 
                if idx not in pure_idxs or idx == raw_idxs[0]]
        perif_spots.append(gsva.index[idxs[:n_nbrs]])
    
    perif_dict = {}
    for (spots, ct) in zip(perif_spots, gsva.columns):
        for spot in spots:
            if spot not in perif_dict.keys():
                perif_dict[spot] = [ct]
            else:
                perif_dict[spot].append(ct)
        
    pure_spots = np.asarray(pure_spots)
    perif_spots = np.asarray(perif_spots).flatten()
    
    adata.obs['pure_val'] = [
        pure_dict[spot] 
        if spot in pure_spots else 'nan' for spot in adata.obs_names 
    ]
    adata.obs['perif_val'] = [
        perif_dict[spot]
        if spot in perif_spots else [np.nan] for spot in adata.obs_names 
    ]
    adata.obs['perif_unique_val'] = adata.obs['perif_val'].apply(lambda x: x[0])

    return pure_spots, perif_spots, adata
    
    
def calc_priors(adata, sig_genes_dict):
    """Calculate avg. gexp variance for each signature as priors"""
    # Subset signature genes as Union(marker_genes, highly_variable genes)
    marker_genes = list(set(
        [gene
         for genes in sig_genes_dict.values() 
         for gene in genes]
    ))
    hv_genes = adata.var_names[adata.var['highly_variable']]
    sel_genes = set(np.union1d(marker_genes, hv_genes))
    genes_dict = {}
    for ct, genes in sig_genes_dict.items():
        filtered_genes = [gene for gene in genes if gene in sel_genes]
        if len(filtered_genes) > 0:
            genes_dict[ct] = filtered_genes
            
    # Calculate signature variance priors
    sig_vars = []
    sig_vars_dict = {} 
    
    for ct, genes in genes_dict.items():
        spot_mask = adata.obs['perif_val'].apply(
            lambda x: ct in x
        ).astype(bool)
        gene_mask = adata.var_names.intersection(genes)
        ct_vars = adata[spot_mask, gene_mask].X.sum(axis=1).var()
        sig_vars.append(ct_vars)
        sig_vars_dict[ct] = ct_vars
        
    sig_vars = np.asarray(sig_vars)
    
    # Calculate signature mean priors
    sig_mean = []
    sig_mean_dict = {} 
    
    for ct, genes in genes_dict.items():
        spot_mask = adata.obs['perif_val'].apply(
            lambda x: ct in x
        ).astype(bool)
        gene_mask = adata.var_names.intersection(genes)
        ct_mean = adata[spot_mask, gene_mask].X.sum(axis=1).mean()
        sig_mean.append(ct_mean)
        sig_mean_dict[ct] = ct_mean
        
    sig_mean = np.asarray(sig_mean)
    
    return sig_vars, sig_vars_dict, sig_mean, sig_mean_dict, sel_genes



def save_loss_plot(train_loss, valid_loss,sample_id):
    # loss plots,
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("../results/02_PoE_factor/"+sample_id+'loss.jpg')
    plt.show()