import os
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import sys
IN_COLAB = "google.colab" in sys.modules
if not IN_COLAB:
    import histomicstk as htk
from skimage import io
import cv2
import json

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


def preprocess_img(data_path,sample_id, adata_index, hchannal=False):
    """
    Parameters:
    1.data_path 
    2.sample_id
    
    get hematoxylin channel of the image, and get the loc of image
    """
    if hchannal:
        adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial','hematoxylin.png'))
    else:
        adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial','tissue_hires_image.png'))
        #adata_image = (adata_image-adata_image.min())/(adata_image.max()-adata_image.min())
        adata_image_norm = (adata_image*255).astype(np.uint8)
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stains
        # create stain matrix
        # create stain matrix
        W = np.array([stain_color_map[st] for st in stains]).T
        # perform standard color deconvolution
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(adata_image_norm, W)

        adata_image_h = imDeconvolved.Stains[:,:,0]
        adata_image_e = imDeconvolved.Stains[:,:,2]
        adata_image_h = ((adata_image_h - adata_image_h.min())/(adata_image_h.max()-adata_image_h.min()) *255).astype(np.uint8)
        adata_image_e = ((adata_image_e - adata_image_e.min())/(adata_image_e.max()-adata_image_e.min()) *255).astype(np.uint8)

        #clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(20,20))
        #clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10,10)) 2021-01-05
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))

        adata_image_h = clahe.apply(adata_image_h)
        adata_image_e = clahe.apply(adata_image_e)

        adata_image_h = (adata_image_h - adata_image_h.min())/(adata_image_h.max()-adata_image_h.min())
        adata_image_e = (adata_image_e - adata_image_e.min())/(adata_image_e.max()-adata_image_e.min())
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
    
    return adata_image, adata_image_h, adata_image_e, map_info


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


def load_adata(data_folder,sample_id, n_genes):
    """
    load visium adata, with raw counts, and filtered gene sets
    input: 
        sample_id: the folder name for the data
        n_genes: the number of the gene for training
    """
    if sample_id.startswith('MBC'):
        adata_sample = sc.read_visium(path=os.path.join(root_dir,data_folder, sample_id),library_id =  sample_id)
        adata_sample.var_names_make_unique()
        adata_sample.obs['sample']=sample_id
    elif sample_id.startswith('simu'):
        adata_sample = sc.read_csv(os.path.join(data_folder, sample_id,'counts.st_synth.csv'))

    else:
        adata_sample = sc.read_h5ad(os.path.join(data_folder,sample_id, sample_id+'.h5ad'))
        adata_sample.var_names_make_unique()
        adata_sample.obs['sample']=sample_id
    adata_sample_pre = preprocess(adata_sample, n_top_genes=n_genes) 
    adata_sample = adata_sample[:,adata_sample_pre.var_names]
    adata_sample.var = adata_sample_pre.var
    adata_sample.obs = adata_sample_pre.obs
    return adata_sample

def get_adata_wsig(adata_sample, gene_sig):
    """
    to make sure gene sig in trained gene set
    """
    variable_gene, sig_gname_filtered, sig_variable_gene_inter = gene_for_train(adata_sample, gene_sig)
    adata_sample_filter = adata_sample[:,sig_variable_gene_inter]
    
    return adata_sample_filter


def get_sig_mean(adata_sample, gene_sig):
    sig_version = 'raw' # or log
    gene_sig_exp_m = pd.DataFrame()
    for i in range(gene_sig.shape[1]):
        if sig_version == 'raw':
            gene_sig_exp_m[gene_sig.columns[i]]=adata_sample.to_df().loc[:,np.intersect1d(adata_sample.var_names,np.unique(gene_sig.iloc[:,i].astype(str)))].mean(axis=1)
        else: 
            gene_sig_exp_m[gene_sig.columns[i]]=adata_sample_pre.to_df().loc[:,np.intersect1d(adata_sample_pre.var_names,np.unique(gene_sig.iloc[:,i].astype(str)))].mean(axis=1)
    return gene_sig_exp_m


def get_anchor_spots(adata_sample,
                     sig_mean,
                     v_low = 20, 
                     v_high = 95, 
                     n_anchor = 40 
                    ):
    """
    input: 
        adata_sample: anndata
        v_low: the low threshold
        v_high: the high threshold
        n_anchor: number of anchor spots 
    """
    highq_spots = (((adata_sample.to_df()>0).sum(axis=1)>np.percentile((adata_sample.to_df()>0).sum(axis=1), v_low)) & 
               ((adata_sample.to_df()).sum(axis=1)>np.percentile((adata_sample.to_df()).sum(axis=1), v_low)) & 
               ((adata_sample.to_df()>0).sum(axis=1)<np.percentile((adata_sample.to_df()>0).sum(axis=1), v_high)) & 
               ((adata_sample.to_df()).sum(axis=1)<np.percentile((adata_sample.to_df()).sum(axis=1), v_high)) 
              )
    pure_spots = np.transpose(sig_mean.loc[highq_spots,:].index[(-np.array(sig_mean.loc[highq_spots,:])).argsort(axis=0)[:40,:]])

    pure_dict = {
            ct: spot 
            for (spot, ct) in zip(pure_spots, sig_mean.columns)
        }

    perif_dict = pure_dict

    adata_pure = np.zeros([adata_sample.n_obs,1])
    adata_pure_idx = [np.where(adata_sample.obs_names == i)[0][0] for i in sorted({x for v in perif_dict.values() for x in v})]
    adata_pure[adata_pure_idx]=1
    return pure_spots, pure_dict, adata_pure



def get_umap(adata_sample):
    
    sc.tl.pca(adata_sample, svd_solver='arpack')
    sc.pp.neighbors(adata_sample, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_sample,  min_dist=0.2)
    sc.pl.umap(adata_sample)
    umap_plot = pd.DataFrame(adata_sample.obsm['X_umap'],
                         columns=['umap1','umap2'],
                         index = adata_sample.obs_names)
    return umap_plot

def get_simu_map_info(umap_plot):
    map_info = []
    map_info = [-umap_plot['umap2']*10,umap_plot['umap1']*10]
    map_info = pd.DataFrame(np.transpose(map_info),
                            columns=['array_row','array_col'],
                            index = umap_plot.index)
    return map_info

def get_windowed_library(adata_sample,map_info,library, window_size):
    library_n = []
    for i in adata_sample.obs_names:
        window_size = window_size
        dist_arr = np.sqrt((map_info.loc[:,'array_col']-map_info.loc[i,'array_col'])**2 + (map_info.loc[:,'array_row']-map_info.loc[i,'array_row'])**2)
        dist_arr<window_size
        library_n.append(library[dist_arr<window_size].mean())
    library_n = np.array(library_n)
    return library_n