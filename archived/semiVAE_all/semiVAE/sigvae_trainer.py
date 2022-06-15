import numpy as np
import pandas as pd
import torch

from .utils import load_visium

def train(model,
          adata_train,
          mu_gexp,
          df_gsva_train,
          pseudo_spots,
          batch_size=128,
          n_epochs=400,
          lr=1e-3,
          alpha=0.60,
          beta=0.35
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

    mu_gexp : np.ndarray
        Mean signature gexp of pseudo-spots for each factor, [S' x D]
        (*S': pseudo-spots dim., D: signature dim.)

    df_gsva_train : pd.DataFrame
        Discretized GSVA score for training, [S x D]
        (*D: signature dim.)

    pseudo_spots : list
        List of signature-specific peripheral (pseudo) spots close to the purest spots representing each signature

    alpha : float
        dynamic weight to adjust anchor loss regularization

    beta : float
        dynamic weight to adjust signature loss regularization

    Returns
    -------
    losses : list
        Log of loss values during training
    """
    dataloader = load_visium(adata_train, batch_size=batch_size)
    adata_ps = adata_train[pseudo_spots, :]
    x_sample = torch.Tensor(adata_ps.X) if isinstance(adata_ps.X, np.ndarray) else torch.Tensor(adata_ps.X.A)
    mu_gexp = torch.Tensor(mu_gexp)
    gsva_sig = torch.Tensor(df_gsva_train.loc[pseudo_spots, :].to_numpy())

    assert mu_gexp.shape == gsva_sig.shape, \
        "Inconsistent dimension between mean signature gexp. & signature GSVA score"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mu_gexp = mu_gexp.to(device)
    x_sample = x_sample.to(device)
    gsva_sig = gsva_sig.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience = 10
    losses = []
    min_sigloss = np.inf

    for epoch in range(n_epochs):
        if patience == 0:
            beta = min(0.999-alpha, beta+0.001)

        if epoch % 100 == 0:
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loss_dict = run_one_epoch(
            model=model,
            dataloader=dataloader,
            x_sample=x_sample,
            alpha=alpha,
            beta=beta,
            mu_gexp=mu_gexp,
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
            print("[Epoch %i]: NLL=%.2f; KL=%.2f; Anchor=%.2f; Sig=%.2f; Total Loss=%.2f; α=%.3f, β=%.3f" %
                  (epoch + 1,
                   loss_dict['nll'],
                   loss_dict['kl'],
                   loss_dict['anchor'],
                   loss_dict['sig'],
                   loss_dict['total'],
                   alpha, beta))

    return losses


def run_one_epoch(model,
                  dataloader,
                  x_sample,
                  alpha=None,
                  beta=None,
                  mu_gexp=None,
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
    anchor_loss = []
    sig_loss = []
    total_loss = []

    for x in dataloader:
        x = x.float()
        x = x.to(device)
        encoded, decoded, latent = model(x, x_sample)

        loss_dict = model.loss(
            x=x,
            mu_gexp=mu_gexp,
            gsva_sig=gsva_sig,
            alpha=alpha,
            beta=beta
        )

        loss = loss_dict['total']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        nll.append(loss_dict['nll'].detach().cpu().numpy().mean())
        kl.append(loss_dict['kl'].detach().cpu().numpy().mean())
        anchor_loss.append(loss_dict['anchor'].detach().cpu().numpy().mean())
        sig_loss.append(loss_dict['sig'].detach().cpu().numpy())
        total_loss.append(loss_dict['total'].detach().cpu().numpy())

    return {
        'nll': np.mean(nll),
        'kl': np.mean(kl),
        'anchor' : np.mean(anchor_loss),
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
