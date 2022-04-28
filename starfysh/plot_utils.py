import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


def plot_anchor_spots(umap_plot,
                      pure_spots,
                      sig_mean):
    fig,ax = plt.subplots(1,1,dpi=200,figsize=(2,2))
    ax.scatter(umap_plot['umap1'],
               umap_plot['umap2'],
               s=2,
               color='lightgray')
    for i in range(len(pure_spots)):
        ax.scatter(umap_plot['umap1'][pure_spots[i]],
                   umap_plot['umap2'][pure_spots[i]],
                   s=20)
    plt.legend(['all']+[i for i in sig_mean.columns],
               loc='right', 
               bbox_to_anchor=(2.2,0.5),)
    ax.grid(False)
    ax.axis('off')
    
def plot_proportions(umap_df,proportions,idx, cmap = 'viridis'):
    
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
