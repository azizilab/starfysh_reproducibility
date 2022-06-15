import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.colors as mcolors
import seaborn as sns
import scanpy as sc

from scipy.stats import gaussian_kde

from .utils import calc_r2, calc_diag_score, get_marker_genes


def _get_color_names(size=10):
    """Get list of color names"""
    assert size <= 10, "Insufficient number of colors"

    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                    for name, color in mcolors.TABLEAU_COLORS.items())
    names = [name for hsv, name in by_hsv]
    return names[:size]


def display_reconst(df_true,
                    df_pred,
                    density=False,
                    marker_genes=None,
                    sample_rate=0.1,
                    size=(20, 20),
                    spot_size=15,
                    title=None
                    ):
    """
    Scatter plot - raw gexp vs. reconstructed gexp
    """
    assert 0 < sample_rate <= 1, \
        "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

    if marker_genes is not None:
        marker_genes = set(marker_genes)

    df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
    df_pred_sample = df_pred.loc[df_true_sample.index]

    plt.rcParams["figure.figsize"] = size
    ax = plt.gca()

    xx = df_true_sample.T.to_numpy().flatten()
    yy = df_pred_sample.T.to_numpy().flatten()

    if density:
        for gene in df_true_sample.columns:
            try:
                gene_true = df_true_sample[gene].values
                gene_pred = df_pred_sample[gene].values
                gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

                z = gaussian_kde(gexp_stacked)(gexp_stacked)
                ax.scatter(gene_true, gene_pred, c=z, s=spot_size, alpha=0.5)
            except np.linalg.LinAlgError as e:
                pass

    elif marker_genes is not None:
        color_dict = {True: 'red', False: 'green'}
        gene_colors = np.vectorize(
            lambda x: color_dict[x in marker_genes]
        )(df_true_sample.columns)
        colors = np.repeat(gene_colors, df_true_sample.shape[0])

        ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

    else:
        ax.scatter(xx, yy, s=spot_size, alpha=0.5)

    min_val = min(xx.min(), yy.min())
    max_val = max(xx.max(), yy.max())
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.suptitle(title)
    plt.xlabel('Ground-truth')
    plt.ylabel('Reconstructed')

    plt.show()


def display_gexp_var(var_true, var_pred, title):
    """
    Scatter plot - Ground-Truth vs. predicted gene-specific variance
    """
    r_2 = round(calc_r2(var_true, var_pred), 2)

    plt.figure(figsize=(10, 10))
    plt.scatter(var_true, var_pred)

    plt.xlabel('Ground-Truth')
    plt.ylabel('Reconstructed')
    plt.suptitle(title + ' $R^2$={}'.format(str(r_2)))
    plt.show()


def display_corr_gsva(df_corr,
                      cluster=False,
                      title=None,
                      size=(12, 20)
                      ):
    """
    Correlation heatmap - Bottle-neck (latent) values vs. GSVA score

    Parameters
    ----------
    df_corr : pd.DataFrame
        DataFrame of correlation between factors & learnt latent variable

    cluster : bool [default=False]
        Whether perform hierarchical clustering on axes
    """
    # Performance metrics: F1-score as diagonalizedness measurement
    score = str(round(calc_diag_score(df_corr), 2))

    # Plotting specs
    sns.set(font_scale=1.5)
    cmap = colormap.get_cmap('RdBu')

    g = sns.clustermap(df_corr, cmap=cmap.reversed(), row_cluster=cluster, col_cluster=cluster)
    g.fig.set_size_inches(size[0], size[1])
    ax = g.ax_heatmap

    ax.set_xticks(np.arange(len(df_corr.columns)))
    ax.set_yticks(np.arange(len(df_corr.index)))
    ax.set_xticklabels(df_corr.columns)
    ax.set_yticklabels(df_corr.index)

    plt.suptitle(title + '; F1-score={}'.format(score))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.show()


def display_latent(adata, df_gsva, model, deconv=True, vmin=0, vmax=0.3, cmap='Purples'):
    """Display VAE decomposition results learnt by latent BN layer for each signature"""
    assert cmap in plt.colormaps() or isinstance(cmap, mcolors.ListedColormap), \
        "Invalid colormap {}".format(cmap)
    signatures = df_gsva.columns
    decomp = model.get_deconvolution().T if deconv else model.qz_mu.detach().cpu().numpy().T
    for sig, z in zip(signatures, decomp):
        adata.obs[sig] = z

    sc.set_figure_params(scanpy=True, fontsize=14)
    sc.pl.spatial(adata, img_key='hires', color=signatures, ncols=3,
                  vmin=vmin, vmax=vmax, alpha_img=0.5, cmap=cmap)


def display_sig_gexp(adata, df_gsva, sig_genes_dict, vmin=0, vmax=3, cmap='seismic'):
    """Display average gexp of each signature gene sets"""
    assert cmap in plt.colormaps() or isinstance(cmap, mcolors.ListedColormap), \
        "Invalid colormap {}".format(cmap)
    marker_genes = get_marker_genes(adata, sig_genes_dict)
    filtered_sig_genes_dict = {
        sig: np.intersect1d(genes, marker_genes)
        for sig, genes in sig_genes_dict.items() if len(np.intersect1d(genes, marker_genes)) > 0
    }
    signatures = df_gsva.columns
    for sig in signatures:
        key = sig + '_gexp'
        adata_gexp = adata[:, filtered_sig_genes_dict[sig]]
        avg_gexp = adata_gexp.X.mean(axis=1) if (adata_gexp.X, np.ndarray) else adata_gexp.X.A.mean(axis=1)
        adata[key] = avg_gexp

    sc.set_figure_params(scanpy=True, fontsize=14)
    sc.pl.spatial(adata, img_key='hires', color=signatures, ncols=3,
                  vmin=vmin, vmax=vmax, alpha_img=0.5, cmap=cmap)
