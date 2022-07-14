import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import r2_score


def plot_anchor_spots(umap_plot,
                      pure_spots,
                      sig_mean,
                      bbox_x=2,
                     ):
    fig,ax = plt.subplots(1,1,dpi=300,figsize=(3,3))
    ax.scatter(umap_plot['umap1'],
               umap_plot['umap2'],
               s=2,
               alpha=1,
               color='lightgray')
    for i in range(len(pure_spots)):
        ax.scatter(umap_plot['umap1'][pure_spots[i]],
                   umap_plot['umap2'][pure_spots[i]],
                   s=8)
    plt.legend(['all']+[i for i in sig_mean.columns],
               loc='right', 
               bbox_to_anchor=(bbox_x,0.5),)
    ax.grid(False)
    ax.axis('off')


def plot_sp_anchor_spots(map_info,
                      pure_spots,
                      sig_mean,
                      bbox_x=2,
                     ):
    fig,ax = plt.subplots(1,1,dpi=200,figsize=(2,2))
    ax.scatter(map_info['array_col'],
               -map_info['array_row'],
               s=2,
               color='lightgray')
    for i in range(len(pure_spots)):
        ax.scatter(map_info['array_col'][pure_spots[i]],
                   -map_info['array_row'][pure_spots[i]],
                   s=20)
    plt.legend(['all']+[i for i in sig_mean.columns],
               loc='right', 
               bbox_to_anchor=(bbox_x,0.5),)
    ax.grid(False)
    ax.axis('off')


def plot_proportions(umap_df, proportions, idx, cmap='viridis'):
    fig,ax = plt.subplots(1,1,dpi=300,figsize=(2.5,2))
    plt.scatter(umap_df['umap1'],
                umap_df['umap2'],
                s=1,
                cmap=cmap,
                c=np.array(proportions.iloc[:,idx])
               )
    plt.title(proportions.columns[idx])
    plt.colorbar()
    ax.grid(False)
    ax.axis('off')


def plot_corr(y_true, y_pred, outdir=None, filename=None, is_sig_mean=False, savefig=False):
    """
    Calculate & plot correlation of cell proportion (or absolute cell abundance)
    between ground-truth & predictions (both [S x F])
    """
    assert y_true.shape == y_pred.shape, 'Inconsistent dimension between ground-truth & prediction'

    v1, v2 = y_true.values, y_pred.values
    n_factors = v1.shape[1]
    corr = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        for j in range(n_factors):
            corr[i, j], _ = np.round(pearsonr(v1[:, i], v2[:, j]), 4)

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
    g = sns.heatmap(corr, annot=True,
                    cmap='RdBu_r', vmin=-1, vmax=1,
                    cbar_kws={'label': 'Cell type proportion corr.'},
                    ax=ax
                    )

    ax.set_xticks(np.arange(n_factors) + 0.5, labels=y_pred.columns, rotation=90)
    ax.set_yticks(np.arange(n_factors) + 0.5, labels=y_true.columns, rotation=0)
    if is_sig_mean:
        ax.set_xlabel('Signature Mean')
    else:
        ax.set_xlabel('Estimated proportion')
    ax.set_ylabel('Ground truth proportion')

    plt.show()

    if savefig and outdir is not None:
        if filename is None:
            filename = 'corr'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, filename + '.eps'), transparent=True, bbox_indches='tight', format='eps')

    return fig, ax


def plot_prop_scatter(y_true, y_pred, outdir=None, filename=None, savefig=False):
    """
    Scatter plot of spot-wise proportion between ground-truth & predictions
    """

    assert y_true.shape == y_pred.shape, 'Inconsistent dimension between ground-truth & prediction'

    n_factors = y_true.shape[1]
    y_true_vals = y_true.values
    y_pred_vals = y_pred.values

    fig, axes = plt.subplots(1, n_factors, figsize=(2 * n_factors, 2.2), dpi=300)

    for i, ax in enumerate(axes):
        v1 = y_true_vals[:, i]
        v2 = y_pred_vals[:, i]
        v_stacked = np.vstack([v1, v2])
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
        ax.annotate(r"$R^2$ = {:.3f}".format(r2_score(v1, v2)), (0.0, 0.9), fontsize=8)

        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        ax.set_xticks(np.arange(0, 1.1, 0.5))
        ax.set_yticks(np.arange(0, 1.1, 0.5))

        ax.set_xlabel('Ground truth proportions')
        ax.set_ylabel('Predicted proportions')

    plt.tight_layout()
    plt.show()

    if savefig and outdir is not None:
        if filename is not None:
            filename = 'proportion_scatter'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig(os.path.join(outdir, filename + '.eps'), transparent=True, bbox_indches='tight', format='eps')

    return fig, axes


def plot_spatial_var(map_info,
                     adata_sample,
                     log_lib,
                     label
                     ):
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])
    fig,axs= plt.subplots(1,1,figsize=(2.5,2),dpi=300)
    g=axs.scatter(all_loc[:,0],
                  -all_loc[:,1],
                  c=log_lib,
                  cmap='magma',
                  s=1
                 )
    fig.colorbar(g,label=label)
    plt.axis('off')

    
def pl_spatial_feature(adata_sample,
               map_info,
               feature, 
               idx,
               plt_title,
               label,
               vmin=None,
               vmax=None,
               s=3,      
              ):
    qvar = feature
    color_idx_list = (qvar[:,idx].astype(float))
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])
    fig,axs= plt.subplots(1,1,figsize=(4,3),dpi=200)
    g=axs.scatter(all_loc[:,0],-all_loc[:,1],cmap='magma',c=color_idx_list,s=s,vmin=vmin,vmax=vmax)

    #plt.legend(color_unique_idx,title='disc_gsva',loc='right',bbox_to_anchor=(1.3, 0.5))
    fig.colorbar(g,label=label)
    plt.title(plt_title)
    axs.set_xticks([])
    axs.set_yticks([])
    plt.axis('off')
    
    pass


def pl_umap_feature(adata_sample,
               map_info,
               feature, 
               idx,
               plt_title,
               label,
               vmin=None,
               vmax=None,
               s=3,      
              ):
    qvar = feature
    color_idx_list = (qvar[:,idx].astype(float))
    all_loc = np.array(map_info.loc[:,['umap1','umap2']])
    fig,axs= plt.subplots(1,1,figsize=(4,3),dpi=200)
    g=axs.scatter(all_loc[:,0],all_loc[:,1],cmap='Spectral_r',c=color_idx_list,s=s,vmin=vmin,vmax=vmax)

    #plt.legend(color_unique_idx,title='disc_gsva',loc='right',bbox_to_anchor=(1.3, 0.5))
    fig.colorbar(g,label=label)
    plt.title(plt_title)
    axs.set_xticks([])
    axs.set_yticks([])
    plt.axis('off')
    
    pass


def pl_spatial_inf_feature(adata_sample,
               map_info,
               inference_outputs,
               feature, 
               idx,
               plt_title,
               label,
               vmin=None,
               vmax=None,
               s=3,      
              ):
    qvar = inference_outputs[feature].detach().cpu().numpy()
    color_idx_list = ((qvar[:,idx].astype(float)))
    #color_idx_list = (color_idx_list-color_idx_list.min())/(color_idx_list.max()-color_idx_list.min())
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])
    fig,axs= plt.subplots(1,1,figsize=(4,3.5),dpi=200)
    g=axs.scatter(all_loc[:,0],-all_loc[:,1],cmap='Blues',c=color_idx_list,s=s,vmin=vmin,vmax=vmax)

    #plt.legend(color_unique_idx,title='disc_gsva',loc='right',bbox_to_anchor=(1.3, 0.5))
    fig.colorbar(g,label=label)
    plt.title(plt_title)
    axs.set_xticks([])
    axs.set_yticks([])
    plt.axis('off')
    
    pass