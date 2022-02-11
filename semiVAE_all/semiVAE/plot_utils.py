import numpy as np
import matplotlib.pyplot as plt


def pl_spatial_cluster(adata_sample,
               map_info,
               s=3
               
              ):
    color_idx_list = list(adata_sample.obs['clusters'].astype(int))
    color_unique_idx = (adata_sample.obs['clusters'].astype(int)).unique()
    all_loc = np.array(map_info.loc[:,['array_col','array_row']])

    fig,axs= plt.subplots(1,1,figsize=(4,4),dpi=200)
    for i in range(len((adata_sample.obs['clusters'].astype(int)).unique())):
        g=axs.scatter(all_loc[color_idx_list==color_unique_idx[i],0],-all_loc[color_idx_list==color_unique_idx[i],1],s=3)

    plt.legend(color_unique_idx,loc='right',bbox_to_anchor=(1.3, 0.5))
    axs.set_xticks([])
    axs.set_yticks([])
    
    pass