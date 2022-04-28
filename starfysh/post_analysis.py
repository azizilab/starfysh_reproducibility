import umap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

def get_z_umap(inference_outputs):
    qz_m_ct = inference_outputs["qz_m_ct"].detach().numpy()
    
    fit = umap.UMAP(
            n_neighbors=45,
            min_dist=0.5,
                   )
    u = fit.fit_transform(qz_m_ct.reshape([2551,-1]))    
    return u


def plot_type_all(inference_outputs,u, proportions):
    qc_m = inference_outputs["qc_m"].detach().numpy()
    group_c = np.argmax(qc_m,axis=1)
    plt.figure(dpi=500,figsize=(2,2))
    cmaps = ['Blues','Greens','Reds','Oranges','Purples']
    for i in range(5):
        plt.scatter(u[group_c==i,0],u[group_c==i,1],s=1,c = qc_m[group_c==i,i], cmap=cmaps[i])
    plt.legend(proportions.columns,loc='right', bbox_to_anchor=(2.2,0.5),)
    #plt.colorbar(label='cell number')
    #plt.title('UMAP of z')
    plt.axis('off')
    
    
def get_corr_map(inference_outputs,sig_mean,proportions):
    qc_m_n = inference_outputs["qc_m"].detach().numpy()
    corr_map_qcm = np.zeros([qc_m_n.shape[1],qc_m_n.shape[1]])
    corr_map_genesig = np.zeros([qc_m_n.shape[1],qc_m_n.shape[1]])
    for i in range(corr_map_qcm.shape[0]):
        for j in range(corr_map_qcm.shape[0]):
            corr_map_qcm[i,j], _ = pearsonr(qc_m_n[:,i], proportions.iloc[:,j])
            corr_map_genesig[i,j], _ = pearsonr(sig_mean.iloc[:,i], proportions.iloc[:,j])  

    plt.figure(dpi=300,figsize=(3.2,3.2))
    ax = sns.heatmap(corr_map_qcm, annot=True,
                     cmap='RdBu_r',vmax=1,vmin=-1,
                     cbar_kws={'label': 'Cell type proportion corr.'}
                    )
    #plt.imshow(corr_map_qcm,vmin=-1,vmax=1,cmap='RdBu_r')
    #plt.set_xtickslabel(['1','2','3'])
    plt.xticks([0.5,1.5,2.5,3.5,4.5],labels=proportions.columns,rotation=90)
    plt.yticks([0.5,1.5,2.5,3.5,4.5],labels=proportions.columns,rotation=0)
    plt.xlabel('Estimated proportion')
    plt.ylabel('Ground truth proportion')


    plt.figure(dpi=300,figsize=(3.2,3.2))
    ax = sns.heatmap(corr_map_genesig, annot=True,
                     cmap='RdBu_r',vmax=1,vmin=-1,
                     cbar_kws={'label': 'Cell type proportion corr.'}
                    )
    #plt.imshow(corr_map_genesig,vmin=-1,vmax=1,cmap='RdBu_r')
    #plt.set_xtickslabel(['1','2','3'])

    plt.xticks([0.5,1.5,2.5,3.5,4.5],labels=proportions.columns,rotation=90)
    plt.yticks([0.5,1.5,2.5,3.5,4.5],labels=proportions.columns,rotation=0)
    plt.xlabel('Gene expression mean')
    plt.ylabel('Ground truth proportion')

    
def pl_spatial_feature(adata_sample,
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
    qvar = inference_outputs[feature].detach().numpy()
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



from scipy.stats import gaussian_kde
def display_reconst(df_true,
                    df_pred,
                    density=False,
                    marker_genes=None,
                    sample_rate=0.1,
                    size=(3, 3),
                    spot_size=1,
                    title=None,
                    x_label='',
                    y_label='',
                    x_min=0,
                    x_max=10,
                    y_min=0,
                    y_max=10,
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
    plt.figure(dpi=300)
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
    #ax.set_xlim(min_val, 400)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    #ax.set_ylim(min_val, 400)

    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axis('equal')
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()