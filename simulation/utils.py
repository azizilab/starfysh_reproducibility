import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, gaussian_kde


# -----------------
# Benchmark metrics
# -----------------

def dist2identity(A):
    """
    Calculate the distance to identity matrix
    """
    n = A.shape[0]
    I = np.identity(n)
    return np.linalg.norm(A - I, ord='fro')


def bootstrap_dists(corr_df, n_iter=1000, size=10):
    """
    Calculate the avg. distance to identity matrix based on random subsampling
    """
    size = corr_df.shape[0]
    n = min(size, size)
    labels = corr_df.columns
    dists = np.zeros(n_iter)

    for i in range(n_iter):
        lbl = np.random.choice(corr_df.columns, n)
        A = corr_df.loc[lbl, lbl].values
        dists[i] = dist2identity(A)

    return dists


def calc_colcorr(df1, df2):
    """Calculate per-column correlations between 2 dataframes"""
    assert len(df1.columns) == len(df2.columns), "Dataframes need to be the same to calculate per-col correlations"
    corr = df1.corrwith(df2, axis=0).mean()
    return corr


def get_best_permute(df, prop_true):
    """
    Calculate the best permutation to align reference-free method `factors` to ground-truth cell typess
    """
    nfactors = df.shape[1]
    best_score = -np.inf  # best overall per-column correlations so far
    best_permutation = None
    for col in itertools.permutations(df.columns, nfactors):
        prop_permuted = df[list(col)]
        score = calc_colcorr(prop_true, prop_permuted)
        if score > best_score:
            best_score = score
            best_permutation = col

    df_permuted = df[list(best_permutation)]
    return df_permuted


# -----------------
# Plotting
# -----------------

def disp_corr(y_true, y_pred,
              outdir=None,
              fontsize=5,
              filename=None,
              savefig=False,
              format='png'):
    """
    Calculate & plot correlation of cell proportion (or absolute cell abundance)
    between ground-truth & predictions (both [S x F])
    """

    assert y_true.shape == y_pred.shape, 'Inconsistent dimension between ground-truth & prediction'
    if savefig:
        assert format == 'png' or format == 'eps' or format == 'svg', "Invalid saving format"

    v1 = y_true.values
    v2 = y_pred.values

    n_factors = v1.shape[1]
    corr = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        for j in range(n_factors):
            corr[i, j], _ = np.round(pearsonr(v1[:, i], v2[:, j]), 3)

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=300)
    ax = sns.heatmap(corr, annot=True,
                     cmap='RdBu_r', vmin=-1, vmax=1,
                     annot_kws={"fontsize": fontsize},
                     cbar_kws={'label': 'Cell type proportion corr.'},
                     ax=ax
                    )

    ax.set_xticks(np.arange(n_factors)+0.5)
    ax.set_yticks(np.arange(n_factors)+0.5)
    ax.set_xticklabels(y_pred.columns, rotation=90)
    ax.set_yticklabels(y_true.columns, rotation=0)
    ax.set_xlabel('Estimated proportion')
    ax.set_ylabel('Ground truth proportion')

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)

    if savefig and (outdir is not None and filename is not None):
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, filename+'.'+format), bbox_inches='tight', format=format)

    plt.show()


def disp_prop_scatter(y_true, y_pred, outdir=None, filename=None, savefig=False, format='png'):
    """
    Scatter plot of spot-wise proportion between ground-truth & predictions
    """

    assert y_true.shape == y_pred.shape, 'Inconsistent dimension between ground-truth & prediction'
    if savefig:
        assert format == 'png' or format == 'eps' or format == 'svg', "Invalid saving format"

    n_factors = y_true.shape[1]
    y_true_vals = y_true.values
    y_pred_vals = y_pred.values

    fig, axes = plt.subplots(1, n_factors, figsize=(2 * n_factors, 2.2), dpi=300)

    for i, ax in enumerate(axes):
        v1 = y_true_vals[:, i]
        v2 = y_pred_vals[:, i]
        v_stacked = np.vstack([v1, v2])
        r2 = r2_score(v1, v2)
        den = gaussian_kde(v_stacked)(v_stacked)

        ax.scatter(v1, v2, c=den, s=1, cmap='turbo', vmax=den.max() / 3)

        ax.set_aspect('equal')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axis('equal')

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_title(y_pred.columns[i])
        if i == 0 or i == 1 or i == 2 or i == 4:
            ax.annotate(r"$R^2$ = {:.3f}".format(r2), (0, 1), fontsize=8)
        else:
            ax.annotate(r"$R^2$ = {:.3f}".format(r2), (0.1, 0.9), fontsize=8)

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(np.arange(0, 1.1, 0.5))
        ax.set_yticks(np.arange(0, 1.1, 0.5))

        ax.set_xlabel('Ground truth proportions')
        ax.set_ylabel('Predicted proportions')

    plt.tight_layout()


# TODO: finalize the function to re-plot supp fig.1

"""
e.g.:

g = sns.clustermap(corr_bp_df,
                   figsize=(8, 8),
                   cmap='seismic', 
                   square=True,
                   vmin=-1,
                   vmax=1,
                   xticklabels=True, 
                   yticklabels=True,
                   row_cluster=False,
                   col_cluster=False,
                   annot_kws={'size': 15}
                  )

text = g.ax_heatmap.set_title('BayesPrism vs. Marker gene expressions\nDistance={}'.format(
        np.round(dists_bayesprism.mean(), 3)),
        fontsize=20, x=0.6, y=1.3)
g.savefig('../../plot_files/marker_bayesprism_corr.png', bbox_extra_artists=(text,), bbox_inches='tight', dpi=300)
"""

# TODO: finalize the function to compare F-dist across benchmark methods
"""
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(5, 3))

ax.hist(dists_starfysh, color='blue', alpha=0.2, bins=50, density=True, label='Starfysh')
ax.hist(dists_cell2loc, color='cyan', alpha=0.2, bins=50, density=True, label='Cell2Location')
ax.hist(dists_stdeconv, color='green', alpha=0.2, bins=50, density=True, label='STDeconvolve')
ax.hist(dists_bayesprism, color='red', alpha=0.2, bins=50, density=True, label='BayesPrism')

# plot fitted KDEs
xx_starfysh = np.linspace(dists_starfysh.min(), dists_starfysh.max(), len(dists_starfysh))
xx_cell2loc = np.linspace(dists_cell2loc.min(), dists_cell2loc.max(), len(dists_cell2loc))
xx_stdeconv = np.linspace(dists_stdeconv.min(), dists_stdeconv.max(), len(dists_stdeconv))
xx_bayesprism = np.linspace(dists_bayesprism.min(), dists_bayesprism.max(), len(dists_bayesprism))

kde1 = gaussian_kde(dists_starfysh)
kde2 = gaussian_kde(dists_cell2loc)
kde3 = gaussian_kde(dists_stdeconv)
kde4 = gaussian_kde(dists_bayesprism)

ax.plot(xx_starfysh, kde1(xx_starfysh), color='blue', alpha=0.7)
ax.plot(xx_cell2loc, kde2(xx_cell2loc), color='cyan', alpha=0.7)
ax.plot(xx_stdeconv, kde3(xx_stdeconv), color='green', alpha=0.7)
ax.plot(xx_bayesprism, kde4(xx_bayesprism), color='red', alpha=0.7)

ax.set_xlabel(r"$\Vert A - I \Vert_F$")
ax.set_ylabel('Density')

ax.legend()
ax.set_title('Distribution of Forbenius-norm distances \n over sampled submatrices ')
plt.show()
"""
