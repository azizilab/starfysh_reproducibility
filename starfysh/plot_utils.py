import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


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