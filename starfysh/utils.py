import os
import cv2
import json
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
import sys
# import histomicstk as htk
from skimage import io

from starfysh import AVAE, train


# --------------------------------
# Running starfysh with 3-restart
# --------------------------------

def init_weights(module):
    if type(module) == nn.Linear:
        # torch.nn.init.xavier_normal(module.weight)
        # torch.nn.init.kaiming_normal_(module.weight)
        torch.nn.init.kaiming_uniform_(module.weight)

    elif type(module) == nn.BatchNorm1d:
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def get_alpha_min(sig_mean, pure_dict):
    """Calculate alpha_min for Dirichlet dist. for each factor"""
    alpha_min = 0
    for col_idx in sig_mean.columns:
        if (1 / (sig_mean.loc[pure_dict[col_idx], :] / sig_mean.loc[pure_dict[col_idx], :].sum())[col_idx]).max() > alpha_min:
            alpha_min = (1 / (sig_mean.loc[pure_dict[col_idx], :] / sig_mean.loc[pure_dict[col_idx], :].sum())[col_idx]).max()
    return alpha_min


def run_starfysh(adata,
                 dataloader,
                 gene_sig,
                 sig_mean,
                 win_loglib,
                 pure_dict,
                 pure_idx,
                 n_repeats=3,
                 lr=0.001,
                 epochs=50,
                 patience=10,
                 device=torch.device('cpu'),
                 verbose=True
                 ):
    # TODO: need to pass all parameters!!!!!

    """
    Note: adding early-stopping mechanism evaluated by loss c * loss n:
        - early-stopping with patience=10
        - choose best among 3 rerun
    """

    np.random.seed(0)
    max_patience = patience
    alpha_min = get_alpha_min(sig_mean, pure_dict)

    models = [None] * n_repeats
    losses = []
    loss_c_list = np.repeat(np.inf, n_repeats)

    trainset = dataloader.VisiumDataset(
        adata=adata,
        gene_sig_exp_m=sig_mean,
        adata_pure=pure_idx,
        library_n=win_loglib
    )

    trainloader = DataLoader(
        trainset,
        batch_size=32,
        shuffle=True
    )

    for i in range(n_repeats):
        if verbose:
            print(" ===  Restart Starfysh {0} === \n".format(i + 1))

        patience = max_patience

        best_loss_c = np.inf

        model = AVAE(
            adata=adata,
            gene_sig=gene_sig,
            alpha_min=alpha_min,
            win_loglib=win_loglib
        )
        model = model.to(device)

        loss_dict = {
            'reconst': [],
            'c': [],
            'z': [],
            'n': [],
            'tot': []
        }

        # Initialize model params
        if verbose:
            print('Initializing model parameters...')
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            if patience == 0:
                loss_c_list[i] = best_loss_c
                if verbose:
                    print('Saving best-performing model; Early stopping...')
                break

            result = train(
                model,
                trainloader,
                trainset,
                device,
                optimizer
            )
            torch.cuda.empty_cache()

            loss_tot, loss_reconst, loss_z, loss_c, loss_n, corr_list = result

            if loss_c < best_loss_c:
                patience = max_patience
                models[i] = model
                best_loss_c = loss_c
            else:
                patience -= 1

            torch.cuda.empty_cache()

            loss_dict['tot'].append(loss_tot)
            loss_dict['reconst'].append(loss_reconst)
            loss_dict['z'].append(loss_z)
            loss_dict['c'].append(loss_c)
            loss_dict['n'].append(loss_n)

            if (epoch + 1) % 10 == 0 or patience == 0:
                print(
                    "Epoch[{}/{}], train_loss: {:.4f}, train_reconst: {:.4f}, train_z: {:.4f},train_c: {:.4f},train_n: {:.4f}".format(
                        epoch + 1, epochs, loss_tot, loss_reconst, loss_z, loss_c, loss_n))

        losses.append(loss_dict)

        if epoch:
            print(" === Finished training === \n")

    idx = np.argmin(loss_c_list)
    best_model = models[idx]
    loss = losses[idx]

    return best_model, loss


# -------------------
# Preprocessing & IO
# -------------------

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


def preprocess_img(data_path, sample_id, adata_index, hchannal=False):
    """
    Parameters:
    1.data_path 
    2.sample_id
    
    get hematoxylin channel of the image, and get the loc of image
    """
    if hchannal:
        adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial', 'tissue_hires_image.png'))
        # adata_image = (adata_image-adata_image.min())/(adata_image.max()-adata_image.min())
        adata_image_norm = (adata_image * 255).astype(np.uint8)
        stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
        # specify stains of input image
        stains = ['hematoxylin',  # nuclei stain
                  'eosin',  # cytoplasm stain
                  'null']  # set to null if input contains only two stains
        # create stain matrix
        # create stain matrix
        W = np.array([stain_color_map[st] for st in stains]).T
        # perform standard color deconvolution
        imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(adata_image_norm, W)

        adata_image_h = imDeconvolved.Stains[:, :, 0]
        adata_image_e = imDeconvolved.Stains[:, :, 2]
        adata_image_h = (
                (adata_image_h - adata_image_h.min()) / (adata_image_h.max() - adata_image_h.min()) * 255).astype(
            np.uint8)
        adata_image_e = (
                (adata_image_e - adata_image_e.min()) / (adata_image_e.max() - adata_image_e.min()) * 255).astype(
            np.uint8)

        # clahe = cv2.createCLAHE(clipLimit=20.0, tileGridSize=(20,20))
        # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10,10)) 2021-01-05
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

        adata_image_h = clahe.apply(adata_image_h)
        adata_image_e = clahe.apply(adata_image_e)

        adata_image_h = (adata_image_h - adata_image_h.min()) / (adata_image_h.max() - adata_image_h.min())
        adata_image_e = (adata_image_e - adata_image_e.min()) / (adata_image_e.max() - adata_image_e.min())
    # adata_image = np.clip(adata_image, 0.2, 0.8)
    # adata_image = (adata_image - adata_image.min())/(adata_image.max()-adata_image.min())
    else:
        adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial', 'tissue_hires_image.png'))
        # adata_image = (adata_image-adata_image.min())/(adata_image.max()-adata_image.min())
        adata_image_norm = (adata_image * 255).astype(np.uint8)

    # Mapping images to location

    map_info = []
    f = open(os.path.join(data_path, sample_id, 'spatial', 'scalefactors_json.json', ))
    json_info = json.load(f)
    f.close()
    tissue_hires_scalef = json_info['tissue_hires_scalef']

    tissue_position_list = pd.read_csv(os.path.join(data_path, sample_id, 'spatial', 'tissue_positions_list.csv'),
                                       header=None, index_col=0)
    tissue_position_list = tissue_position_list.loc[adata_index, :]
    map_info = tissue_position_list.iloc[:, 1:3]
    map_info.columns = ['array_row', 'array_col']
    map_info.loc[:, 'imagerow'] = tissue_position_list.iloc[:, 3] * tissue_hires_scalef
    map_info.loc[:, 'imagecol'] = tissue_position_list.iloc[:, 4] * tissue_hires_scalef
    map_info.loc[:, 'sample'] = sample_id

    # return adata_image, adata_image_h, adata_image_e, map_info
    return adata_image, map_info


def gene_for_train(adata, df_sig):
    """find the varibale gene name, the mattered gene signatures, and the combined variable+signature for training"""

    variable_gene = adata.var_names[adata.var['highly_variable']]
    print('the number of original variable genes in the dataset', variable_gene.shape)
    print('the number of siganture genes in the dataset', np.unique(df_sig.values.flatten().astype(str)).shape)

    # filter out some genes in the signature not in the var_names
    sig_gname_filtered = np.intersect1d(adata.var_names, np.unique(df_sig.values.flatten().astype(str)))
    print('after filter out some genes in the signature not in the var_names ...', sig_gname_filtered.shape)
    # filter out some genes not highly expressed in the signature
    sig_gname_filtered = sig_gname_filtered[adata.to_df().loc[:, sig_gname_filtered].sum() > 0]
    print('after filter out some genes not highly expressed in the signature ...', sig_gname_filtered.shape)

    sig_variable_gene_inter = set([*np.array(variable_gene), *sig_gname_filtered])
    print('combine the varibale and siganture, the total unique gene number is ...', len(sig_variable_gene_inter))

    return list(variable_gene), list(sig_gname_filtered), list(sig_variable_gene_inter)


def load_adata(data_folder, sample_id, n_genes):
    """
    load visium adata, with raw counts, and filtered gene sets
    input: 
        sample_id: the folder name for the data
        n_genes: the number of the gene for training
    """
    if sample_id.startswith('MBC'):
        adata_sample = sc.read_visium(path=os.path.join(data_folder, sample_id), library_id=sample_id)
        adata_sample.var_names_make_unique()
        adata_sample.obs['sample'] = sample_id
    elif sample_id.startswith('simu'):
        adata_sample = sc.read_csv(os.path.join(data_folder, sample_id, 'counts.st_synth.csv'))

    else:
        adata_sample = sc.read_h5ad(os.path.join(data_folder, sample_id, sample_id + '.h5ad'))
        adata_sample.var_names_make_unique()
        adata_sample.obs['sample'] = sample_id
    adata_sample_pre = preprocess(adata_sample, n_top_genes=n_genes)
    adata_sample = adata_sample[:, adata_sample_pre.var_names]
    adata_sample.var = adata_sample_pre.var
    adata_sample.obs = adata_sample_pre.obs

    if '_index' in adata_sample.var.columns:
        adata_sample.var_names = adata_sample.var['_index']

    return adata_sample, adata_sample_pre


def get_adata_wsig(adata_sample, gene_sig):
    """
    to make sure gene sig in trained gene set
    """
    variable_gene, sig_gname_filtered, sig_variable_gene_inter = gene_for_train(adata_sample, gene_sig)
    adata_sample_filter = adata_sample[:, sig_variable_gene_inter]

    return adata_sample_filter


def get_sig_mean(adata_sample, gene_sig, log_lib):
    sig_version = 'raw'  # or log
    gene_sig_exp_m = pd.DataFrame()
    for i in range(gene_sig.shape[1]):
        if sig_version == 'raw':
            adata_df = adata_sample.to_df()
            # adata_df[adata_df == 0] = np.nan
            gene_sig_exp_m[gene_sig.columns[i]] = np.nanmean(
                (adata_df.loc[:, np.intersect1d(adata_sample.var_names, np.unique(gene_sig.iloc[:, i].astype(str)))]),
                axis=1)
        else:
            gene_sig_exp_m[gene_sig.columns[i]] = np.log((adata_sample_pre.to_df().loc[:,
                                                          np.intersect1d(adata_sample_pre.var_names, np.unique(
                                                              gene_sig.iloc[:, i].astype(str)))] + 1)).mean(axis=1)

    # gene_sig_exp_arr = torch.Tensor(np.array(gene_sig_exp_m))
    # gene_sig_exp_arr = gene_sig_exp_arr.detach().cpu().numpy()
    gene_sig_exp_arr = pd.DataFrame(np.array(gene_sig_exp_m), columns=gene_sig_exp_m.columns,
                                    index=adata_sample.obs_names)
    return gene_sig_exp_arr


def filter_gene_sig(gene_sig, adata_df):
    for i in range(gene_sig.shape[0]):
        for j in range(gene_sig.shape[1]):
            gene = gene_sig.iloc[i, j]
            if gene in adata_df.columns:
                if adata_df.loc[:, gene].sum() < 20:
                    gene_sig.iloc[i, j] = 'NaN'
    return gene_sig


def get_anchor_spots(adata_sample,
                     sig_mean,
                     v_low=20,
                     v_high=95,
                     n_anchor=40
                     ):
    """
    input: 
        adata_sample: anndata
        v_low: the low threshold
        v_high: the high threshold
        n_anchor: number of anchor spots 
    """
    highq_spots = (((adata_sample.to_df() > 0).sum(axis=1) > np.percentile((adata_sample.to_df() > 0).sum(axis=1),
                                                                           v_low)) &
                   ((adata_sample.to_df()).sum(axis=1) > np.percentile((adata_sample.to_df()).sum(axis=1), v_low)) &
                   ((adata_sample.to_df() > 0).sum(axis=1) < np.percentile((adata_sample.to_df() > 0).sum(axis=1),
                                                                           v_high)) &
                   ((adata_sample.to_df()).sum(axis=1) < np.percentile((adata_sample.to_df()).sum(axis=1), v_high))
                   )
    pure_spots = np.transpose(
        sig_mean.loc[highq_spots, :].index[(-np.array(sig_mean.loc[highq_spots, :])).argsort(axis=0)[:n_anchor, :]])

    pure_dict = {
        ct: spot
        for (spot, ct) in zip(pure_spots, sig_mean.columns)
    }

    perif_dict = pure_dict

    adata_pure = np.zeros([adata_sample.n_obs, 1])
    adata_pure_idx = [np.where(adata_sample.obs_names == i)[0][0] for i in
                      sorted({x for v in perif_dict.values() for x in v})]
    adata_pure[adata_pure_idx] = 1
    return pure_spots, pure_dict, adata_pure


def get_umap(adata_sample):
    sc.tl.pca(adata_sample, svd_solver='arpack')
    sc.pp.neighbors(adata_sample, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_sample, min_dist=0.2)
    sc.pl.umap(adata_sample)
    umap_plot = pd.DataFrame(adata_sample.obsm['X_umap'],
                             columns=['umap1', 'umap2'],
                             index=adata_sample.obs_names)
    return umap_plot


def get_simu_map_info(umap_plot):
    map_info = []
    map_info = [-umap_plot['umap2'] * 10, umap_plot['umap1'] * 10]
    map_info = pd.DataFrame(np.transpose(map_info),
                            columns=['array_row', 'array_col'],
                            index=umap_plot.index)
    return map_info


def get_windowed_library(adata_sample, map_info, library, window_size):
    library_n = []
    for i in adata_sample.obs_names:
        window_size = window_size
        dist_arr = np.sqrt((map_info.loc[:, 'array_col'] - map_info.loc[i, 'array_col']) ** 2 + (
                map_info.loc[:, 'array_row'] - map_info.loc[i, 'array_row']) ** 2)
        dist_arr < window_size
        library_n.append(library[dist_arr < window_size].mean())
    library_n = np.array(library_n)
    return library_n
