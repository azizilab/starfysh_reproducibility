import os
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import histomicstk as htk
from skimage import io
import cv2
import json
from .dataloader import VisiumDataset

def preprocess(adata_raw, lognorm=True, min_perc=None, max_perc=None, n_top_genes=6000, mt_thld=100):
    """
    author: Yinuo Jin
    Preprocessing ST gexp matrix, remove Ribosomal & Mitochondrial genes
    Parameters
    ----------
    adata_raw : annData
        Spot x Bene raw expression matrix [S x G]
    min_perc : float
        lower-bound percentile of non-zero gexps for filtering spots
    max_perc : float
        upper-bound percentile of non-zero gexps for filtering spots
    n_top_genes: float
        number of the variable genes
    mt_thld : float
        max. percentage of mitochondrial gexps for filtering spots
        with excessive MT expressions
    """
    adata = adata_raw.copy()

    if min_perc and max_perc:
        assert 0 < min_perc < max_perc < 100, \
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
    mask_cell = adata.obs['pct_counts_mt'] < mt_thld
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

    adata = adata[mask_cell, mask_gene]
    
    if lognorm:   
        print('Preprocessing2: Normalize')
        sc.pp.normalize_total(adata, inplace=True) 

        # Preprocessing3: Logarithm
        print('Preprocessing3: Logarithm')
        sc.pp.log1p(adata)
    else:
        print('Skip Normalize and Logarithm')

    # Preprocessing4: Find the variable genes
    print('Preprocessing4: Find the variable genes')
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes, inplace=True)
    

    return adata


def preprocess_img(data_path,sample_id, adata_index):
    """
    Parameters:
    1.data_path 
    2.sample_id
    
    get hematoxylin channel of the image, and get the loc of image
    """
    adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial','tissue_hires_image.png'))
    adata_image = (adata_image*255).astype(np.uint8)
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stains
    # create stain matrix
    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T
    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(adata_image, W)

    adata_image = imDeconvolved.Stains[:,:,0]
    adata_image = ((adata_image - adata_image.min())/(adata_image.max()-adata_image.min()) *255).astype(np.uint8)

    #clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(20,20))
    #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10,10)) 2021-01-05
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))

    adata_image = clahe.apply(adata_image)

    adata_image = (adata_image - adata_image.min())/(adata_image.max()-adata_image.min())
    #adata_image = np.clip(adata_image, 0.2, 0.8)
    #adata_image = (adata_image - adata_image.min())/(adata_image.max()-adata_image.min())
    
    #Mapping images to location
    
    map_info = []
    f = open(os.path.join(data_path, sample_id, 'spatial','scalefactors_json.json',))
    json_info = json.load(f)
    f.close()
    tissue_hires_scalef = json_info['tissue_hires_scalef']
    
    tissue_position_list = pd.read_csv(os.path.join(data_path, sample_id, 'spatial','tissue_positions_list.csv'),header=None, index_col=0)
    tissue_position_list = tissue_position_list.loc[adata_index,:]
    map_info= tissue_position_list.iloc[:,1:3]
    map_info.columns=['array_row','array_col']
    map_info.loc[:,'imagerow'] = tissue_position_list.iloc[:,3] *tissue_hires_scalef
    map_info.loc[:,'imagecol'] = tissue_position_list.iloc[:,4] *tissue_hires_scalef
    map_info.loc[:,'sample'] = sample_id
    
    return adata_image, map_info



def gene_for_train(adata, df_sig):
    """find the varibale gene name, the mattered gene signatures, and the combined variable+signature for training"""
    
    variable_gene = adata.var_names[adata.var['highly_variable']]
    print('the number of original variable genes in the dataset',variable_gene.shape)
    print('the number of siganture genes in the dataset',np.unique(df_sig.values.flatten().astype(str)).shape)
    
    # filter out some genes in the signature not in the var_names
    sig_gname_filtered =  np.intersect1d(adata.var_names,np.unique(df_sig.values.flatten().astype(str)))
    print('after filter out some genes in the signature not in the var_names ...',sig_gname_filtered.shape)
    # filter out some genes not highly expressed in the signature
    sig_gname_filtered = sig_gname_filtered[adata.to_df().loc[:,sig_gname_filtered].sum()>0]
    print('after filter out some genes not highly expressed in the signature ...', sig_gname_filtered.shape)
    
    sig_variable_gene_inter  = set([*np.array(variable_gene) ,*sig_gname_filtered])
    print('combine the varibale and siganture, the total unique gene number is ...', len(sig_variable_gene_inter))
    
    return list(variable_gene), list(sig_gname_filtered), list(sig_variable_gene_inter)

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


def find_spots(adata, gsva, n_nbrs=20):
    """
    Find spots representing `pure spots` & their nearest neighbors
    based on GSVA scores for each cell type
    """
    assert 'highly_variable' in adata.var_keys(), \
        "Please choose highly variable genes first"
    
    # Calculate distance with only highly variable genes
    embedding = adata[:, adata.var['highly_variable']].X#.A
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

    return pure_spots, perif_spots


def calc_var_priors(adata, sig_genes_dict):
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
    
    return sig_vars, sig_vars_dict, sel_genes


def load_visium(adata, batch_size=128):
    """Load ST dataset to VAE model"""
    dataset = VisiumDataset(adata=adata)
    shuffle = True if batch_size < len(adata) else False
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    )
    return dataloader