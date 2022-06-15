import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from .dataset import VisiumDataset

######################
#  IO
######################


def load_gsva(path, file_name, thld=0.7):
    """Load & discretize GSVA scores for training"""
    file_path = os.path.join(path, file_name)
    assert os.path.exists(file_path), "GSVA file doesn't exist!"

    df_gsva = pd.read_csv(file_path, index_col=0)
    df_gsva.columns = df_gsva.columns.str.replace(' ', '')
    df_gsva_train = df_gsva.apply(lambda x: x >= x.quantile(thld), axis=0).astype(np.uint8)

    return df_gsva, df_gsva_train


def load_sig(path, file_name):
    file_path = os.path.join(path, file_name)
    assert os.path.exists(file_path), "Signature gene set file doesn't exist!"

    df_sig = pd.read_csv(file_path)
    sig_genes_dict = {
        sig: df_sig[sig][~pd.isna(df_sig)[sig]].tolist()
        for sig in df_sig.columns
    }

    return sig_genes_dict


def preprocess(adata_raw, min_perc=None, max_perc=None, mt_thld=20):
    """
    Preprocessing ST gexp matrix, remove Ribosomal & Mitochondrial genes

    Parameters
    ----------
    adata_raw : annData
        Spot x Bene raw expression matrix [S x G]

    min_perc : float
        lower-bound percentile of non-zero gexps for filtering spots

    max_perc : float
        upper-bound percentile of non-zero gexps for filtering spots

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
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['rb'] = np.logical_or(
        adata.var_names.str.startswith('RPS'),
        adata.var_names.str.startswith('RPL')
    )

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    mask_cell = adata.obs['pct_counts_mt'] < mt_thld
    mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

    adata = adata[mask_cell, mask_gene]

    return adata


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


######################
# Signature Priors
######################


def get_marker_genes(adata, sig_genes_dict):
    """Get all non-repeated genes from signature gene sets"""
    assert 'highly_variable' in adata.var_keys(), "Please find highly variable genes first!"
    sig_genes = list(set(
        [gene
         for genes in sig_genes_dict.values()
         for gene in genes]
    ))
    hv_genes = adata.var_names[adata.var['highly_variable']]
    marker_genes = np.intersect1d(np.union1d(sig_genes, hv_genes), adata.var_names)

    return marker_genes


def find_spots(adata, df_gsva, n_nbrs=30):
    """Find signature (factor) representative spots and their kNNs based on GSVA score"""
    assert 'highly_variable' in adata.var_keys(), "Please find highly variable genes first!"

    # Calculate distance with only highly variable genes
    adata_hvgs = adata[:, adata.var['highly_variable']]
    embedding = adata_hvgs.X if isinstance(adata_hvgs.X, np.ndarray) else adata_hvgs.X.A
    pure_idxs = np.argmax(df_gsva.values, axis=0)
    pure_spots = df_gsva.idxmax(axis=0)

    pure_dict = {
        spot: ct
        for (spot, ct) in zip(pure_spots, df_gsva.columns)
    }

    nbrs = NearestNeighbors(n_neighbors=n_nbrs + len(pure_spots)).fit(embedding)
    nn_graph = nbrs.kneighbors(embedding)[1]

    perif_spots = []
    perif_idxs = [nn_graph[idx].tolist() for idx in pure_idxs]
    for i, raw_idxs in enumerate(perif_idxs):
        idxs = [idx
                for idx in raw_idxs
                if idx not in pure_idxs or idx == raw_idxs[0]]
        perif_spots.append(df_gsva.index[idxs[:n_nbrs]])

    perif_dict = {}
    for (spots, ct) in zip(perif_spots, df_gsva.columns):
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


def calc_var_priors(adata, sig_genes_dict, return_dict=False):
    """Calculate avg. gexp variance of gene sets from each signature (factor) as VAE priors"""
    assert 'perif_val' in adata.obs_keys(), "Please find pure spots for each signature first!"

    # Filter out genes that are neither in signature gene set nor highly variable in adata
    marker_genes = get_marker_genes(adata, sig_genes_dict)

    # Calculate varaince priors for signature with marker genes
    sig_vars = []
    sig_vars_dict = {}
    for sig, genes in sig_genes_dict.items():
        sel_genes = np.intersect1d(genes, marker_genes)
        if len(sel_genes) > 0:
            spot_mask = adata.obs['perif_val'].apply(lambda x: sig in x).astype(bool)
            gene_mask = adata.var_names.intersection(sel_genes)
            variances = adata[spot_mask, gene_mask].X.sum(axis=1).var()
            sig_vars.append(variances)
            sig_vars_dict[sig] = variances

    return sig_vars, sig_vars_dict if return_dict else sig_vars


######################
# Metrics calculation
######################

def calc_r2(x, y):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    reg = LinearRegression().fit(x, y)
    r2 = reg.score(x, y)
    return r2


def calc_diag_score(A, eps=1e-10):
    """
    Measure accuracy (how diagonal) a correlation matrix is
    Metrics:
     - F1 score: TP / (TP + 1/2(FP + FN))
    """
    A = np.asarray(A)
    A[A < 0] = 0
    tp = np.trace(A)
    fp_fn = A.sum() - tp
    score = tp / (tp + 0.5 * fp_fn + eps)

    return score


def calc_corr_gsva(z, spots, df_gsva):
    """Calculate correlation between VAE latent space (z) & GSVA score of selected spots"""
    df_z = pd.DataFrame(
        z,
        index=spots,
        columns=['bn{}'.format(i + 1) for i in range(z.shape[1])]
    )

    # Initialize correlation matrix, assign value & normalize by row
    df_corr = pd.concat(
        [df_z, df_gsva.loc[spots, :]],
        axis=1,
        keys=['bn', 'gsva']
    ).corr().loc['bn', 'gsva']
    df_corr = df_corr.divide(df_corr.max(axis=1), axis=0)

    return df_corr


def disp_to_var(mu, theta):
    """
    Calculate gene-specific variance from negative binomial mean & dispersion

    Parameters
    ----------
    mu : np.ndarray
        Spot x Gene mean expression [S x G]

    theta : np.ndarray
        Gene-specific inverse dispersion [G]

    Returns
    -------
    variance : np.ndarray
        Gene-specific variance [G]
    """
    if torch.is_tensor(theta):
        theta = theta.detach().cpu().numpy()
    variance = mu.mean(0) + mu.mean(0)**2 / theta
    return variance


######################
# Training
######################


def train(model,
          adata_train,
          df_gsva_train,
          pseudo_spots,
          batch_size=128,
          n_epochs=400,
          lr=1e-3,
          alpha=0.02
          ):
    """
    Parameters
    ----------
    model : SignatureVAE
        VAE model with signature priors

    adata_train : AnnData
        Spot x Gene expression matrix with subset of Union(hv_genes, marker_genes), [S x G']
        hv_genes: highly variable genes
        marker_genes: list of genes in signature gene set

    df_gsva_train : pd.DataFrame
        Spot x Signatures discretized GSVA score for training, [S x D]

    pseudo_spots : list
        List of signature-specific peripheral (pseudo) spots close to the purest spots representing each signature

    alpha : float
        weight to adjust loss calculation between (NLL+KL divergence) &
        signature loss during training

    Returns
    -------
    losses : list
        Log of loss values during training
    """
    adata_ps = adata_train[pseudo_spots, :]
    x_sample = torch.Tensor(adata_ps.X) if isinstance(adata_ps.X, np.ndarray) else torch.Tensor(adata_ps.X.A)
    gsva_sig = torch.Tensor(df_gsva_train.loc[pseudo_spots, :].to_numpy())
    dataloader = load_visium(adata_train, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x_sample = x_sample.to(device)
    gsva_sig = gsva_sig.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience = 10
    losses = []
    min_sigloss = np.inf

    for epoch in range(n_epochs):
        if patience == 0:
            alpha = max(0.001, alpha - 0.001)

        if epoch % 100 == 0:
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loss_dict = run_one_epoch(
            model=model,
            dataloader=dataloader,
            x_sample=x_sample,
            alpha=alpha,
            gsva_sig=gsva_sig,
            optimizer=optimizer,
            device=device
        )

        if loss_dict['sig'] < min_sigloss or epoch == 0:
            min_sigloss = loss_dict['sig']
            patience = 10
        else:
            patience = max(0, patience - 1)

        losses.append(loss_dict)

        if (epoch + 1) % 20 == 0:
            print("[Epoch %i]: NLL=%.2f; KL=%.2f; Sig=%.2f; Total Loss=%.2f; α=%.3f" %
                  (epoch + 1, loss_dict['nll'], loss_dict['kl'], loss_dict['sig'], loss_dict['total'], alpha))

    return losses


def run_one_epoch(model,
                  dataloader,
                  x_sample,
                  alpha=None,
                  gsva_sig=None,
                  optimizer=None,
                  device=torch.device('cuda')
                  ):
    assert optimizer is not None and gsva_sig is not None and alpha is not None, \
        "Must declare optimizer and loss weight (alpha) for training"
    if not next(model.parameters()).is_cuda:
        model.to(device)
    if not x_sample.is_cuda:
        x_sample = x_sample.to(device)
    torch.set_grad_enabled(True)

    nll = []
    kl = []
    sig_loss = []
    total_loss = []

    for x in dataloader:
        x = x.float()
        x = x.to(device)
        encoded, decoded, latent = model(x, x_sample)

        loss_dict = model.loss(
            x=x,
            gsva_sig=gsva_sig,
            alpha=alpha
        )

        loss = loss_dict['total']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nll.append(loss_dict['nll'].detach().cpu().numpy().mean())
        kl.append(loss_dict['kl'].detach().cpu().numpy().mean())
        sig_loss.append(loss_dict['sig'].detach().cpu().numpy())
        total_loss.append(loss_dict['total'].detach().cpu().numpy())

    return {
        'nll': np.mean(nll),
        'kl': np.mean(kl),
        'sig': np.mean(sig_loss),
        'total': np.mean(total_loss)
    }


def run_inference(model, adata, device=torch.device('cuda'), resample=False):
    """
    Feed dataset to the trained model and infer
    latent cell-type decomposition & reconstructed gexp

    Parameters
    ----------
    model : SigVAE
        Trained deconvolution VAE model

    adata : AnnData
        dataset to run one-time inference on

    resample : bool
        return resampled from distribution parameters if True,
        return NN outputs otherwise

    Returns
    -------
    latent : np.ndarray
        Latent variable representing signature decomposition of each spot  [S x D]

    x_hat : np.ndarray
        Reconstructed Spot x Gene expression  [S x G]

    """
    dataloader = load_visium(adata, batch_size=adata.shape[0])
    if not next(model.parameters()).is_cuda:
        model.to(device)
    torch.set_grad_enabled(False)

    x = next(iter(dataloader)).float()
    x = x.to(device)
    x_hat, latent = model.inference(x)

    return x_hat, latent
