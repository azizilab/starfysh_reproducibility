import json
import scanpy as sc
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd
import os
import anndata
import networkx as nx
import sys
import seaborn as sns
sys.path.append('/content/drive/MyDrive/SpatialModelProject/model_test_colab/')
from starfysh import utils

def get_factor_dist(sample_ids,file_path):
    qc_p_dist = {}
    # Opening JSON file
    for sample_id in sample_ids:
        print(sample_id)
        f = open(file_path+sample_id+'_factor.json','r')
        data = json.load(f)   
        qc_p_dist[sample_id] = data['qc_m']
        f.close()
    return qc_p_dist


def get_adata(sample_ids,data_folder):
    adata_sample_all = []
    map_info_all = []
    adata_image_all = []
    for sample_id in sample_ids:
        print('loading...',sample_id)
        if (sample_id.startswith('MBC'))|(sample_id.startswith('CT')):

            adata_sample = sc.read_visium(path=os.path.join(data_folder, sample_id),library_id =  sample_id)
            adata_sample.var_names_make_unique()
            adata_sample.obs['sample']=sample_id
            adata_sample.obs['sample_type']='MBC'
            #adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
            #adata_sample.obs_names  = adata_sample.obs_names+'_'+sample_id
            if '_index' in adata_sample.var.columns:
                adata_sample.var_names=adata_sample.var['_index']
        
        else:
            adata_sample = sc.read_h5ad(os.path.join(data_folder,sample_id, sample_id+'.h5ad'))
            adata_sample.var_names_make_unique()
            adata_sample.obs['sample']=sample_id
            adata_sample.obs['sample_type']='TNBC'
            #adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
            #adata_sample.obs_names  = adata_sample.obs_names+'_'+sample_id
            if '_index' in adata_sample.var.columns:
                adata_sample.var_names=adata_sample.var['_index']

        if data_folder =='simu_data':
            map_info = utils.get_simu_map_info(umap_df)
        else:
            adata_image,map_info = utils.preprocess_img(data_folder,sample_id,adata_sample.obs.index,hchannal=False)
        
        adata_sample.obs_names  = adata_sample.obs_names+'-'+sample_id
        map_info.index = map_info.index+'-'+sample_id
        adata_sample_all.append(adata_sample)  
        map_info_all.append(map_info)  
        adata_image_all.append(adata_image)  
    return adata_sample_all,map_info_all,adata_image_all

def get_Moran(W, X):
    N = W.shape[0]
    term1 = N / W.sum().sum()
    x_m = X.mean()
    term2 = np.matmul(np.matmul(np.diag(X-x_m),W),np.diag(X-x_m))
    term3 = term2.sum().sum()
    term4 = ((X-x_m)**2).sum()
    term5 = term1 * term3 / term4
    return term5

def get_LISA(W, X):
    lisa_score = np.zeros(X.shape)
    N = W.shape[0]
    x_m = X.mean()
    term1 = X-x_m 
    term2 = ((X-x_m)**2).sum()
    for i in range(term1.shape[0]):
        #term3 = np.zeros(X.shape)
        term3 = (W[i,:]*(X-x_m)).sum()
        #for j in range(W.shape[0]):
        #    term3[j]=W[i,j]*(X[j]-x_m)
        #term3 = term3.sum()
        lisa_score[i] = np.sign(X[i]-x_m) * N * (X[i]-x_m) * term3 / term2
        #lisa_score[i] =   N * (X[i]-x_m) * term3 / term2
        
    return lisa_score

def get_SCI(W, X, Y):
    
    N = W.shape[0]
    term1 = N / (2*W.sum().sum())

    x_m = X.mean()
    y_m = Y.mean()
    term2 = np.matmul(np.matmul(np.diag(X-x_m),W),np.diag(Y-y_m))
    term3 = term2.sum().sum()

    term4 = np.sqrt(((X-x_m)**2).sum()) * np.sqrt(((Y-y_m)**2).sum())

    term5 = term1 * term3 / term4

    return term5

def get_cormtx(sample_id, hub_num ):

    prop_i = proportions_df[ids_df['source']==sample_id][cluster_df['cluster']==hub_num]
    loc_i = np.array(map_info_all.loc[prop_i.index].loc[:,['array_col','array_row',]])
    W = np.zeros([loc_i.shape[0],loc_i.shape[0]])

    cor_matrix = np.zeros([gene_sig.shape[1],gene_sig.shape[1]])
    for i in range(loc_i.shape[0]):
        for j in range(i,loc_i.shape[0]):
            if np.sqrt((loc_i[i,0]-loc_i[j,0])**2+(loc_i[i,1]-loc_i[j,1])**2)<=3:
                W[i,j] = 1
                W[j,i] = 1
        #indices = vor.regions[vor.point_region[i]]
        #neighbor_i = np.concatenate([vor.ridge_points[np.where(vor.ridge_points[:,0] == i)],np.flip(vor.ridge_points[np.where(vor.ridge_points[:,1] == i)],axis=1)],axis=0)[:,1]
        #W[i,neighbor_i]=1
        #W[neighbor_i,i]=1
    print('spots in hub ',hub_num, '= ',prop_i.shape[0])
    if prop_i.shape[0]>1:
        for i in range(gene_sig.shape[1]):
            for j in range(i+1,gene_sig.shape[1]):
                    cor_matrix[i,j]=get_SCI(W, np.array(prop_i.iloc[:,i]), np.array(prop_i.iloc[:,j]))
                    cor_matrix[j,i]=cor_matrix[i,j]
    return cor_matrix

def get_hub_cormtx(sample_ids, hub_num):
    cor_matrix = np.zeros([gene_sig.shape[1],gene_sig.shape[1]])
    for sample_id in sample_ids:
        print(sample_id)
        cor_matrix = cor_matrix + get_cormtx(sample_id = sample_id, hub_num=hub_num)
        #print(cor_matrix)
    cor_matrix = cor_matrix/len(sample_ids)
    #cor_matrix = pd.DataFrame(cor_matrix)
    return cor_matrix



def create_corr_network_5(G, node_size_list,corr_direction, min_correlation):
    ##Creates a copy of the graph
    H = G.copy()
    
    ##Checks all the edges and removes some based on corr_direction
    for stock1, stock2, weight in G.edges(data=True):
        #print(weight)
        ##if we only want to see the positive correlations we then delete the edges with weight smaller than 0        
        if corr_direction == "positive":
            ####it adds a minimum value for correlation. 
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        ##this part runs if the corr_direction is negative and removes edges with weights equal or largen than 0
        else:
            ####it adds a minimum value for correlation. 
            ####If correlation weaker than the min, then it deletes the edge
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    
    #crates a list for edges and for the weights
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    
    
    ### increases the value of weights, so that they are more visible in the graph
    #weights = tuple([(0.5+abs(x))**1 for x in weights])
    weights = tuple([x*2 for x in weights])
    #print(len(weights))
    #####calculates the degree of each node
    d = nx.degree(H)
    #print(d)
    #####creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*dict(d).items())
    #import sys, networkx as nx, matplotlib.pyplot as plt

    # Create a list of 10 nodes numbered [0, 9]
    #nodes = range(10)
    node_sizes = []
    labels = {}
    for n in nodelist:
            node_sizes.append( node_size_list[n] )
            labels[n] = 1 * n

    # Node sizes: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

    # Connect each node to its successor
    #edges = [ (i, i+1) for i in range(len(nodes)-1) ]

    # Create the graph and draw it with the node labels
    #g = nx.Graph()
    #g.add_nodes_from(nodes)
    #g.add_edges_from(edges)

    #nx.draw_random(g, node_size = node_sizes, labels=labels, with_labels=True)    
    #plt.show()

    #positions
    positions=nx.circular_layout(H)
    #print(positions)
    
    #Figure size
    plt.figure(figsize=(2,2),dpi=500)

    #draws nodes,
    #options = {"edgecolors": "tab:gray", "alpha": 0.9}
    nx.draw_networkx_nodes(H,positions,
                           #node_color='#DA70D6',
                           nodelist=nodelist,
                           #####the node size will be now based on its degree
                           node_color=_colors['leiden_colors'][hub_num],# 'lightgreen',#pink, 'lightblue',#'#FFACB7',lightgreen B19CD9。#FFACB7 brown
                           alpha = 0.8,
                           node_size=tuple([x**1 for x in node_sizes]),
                           #**options
                           )
    
    #Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=4, 
                            font_family='sans-serif')
    
    ###edge colors based on weight direction
    if corr_direction == "positive":
        edge_colour = plt.cm.GnBu#PiYG_r#RdBu_r#Spectral_r#GnBu#RdPu#PuRd#Blues#PuRd#GnBu OrRd
    else:
        edge_colour = plt.cm.PuRd
        
    #draws the edges
    print(min(weights))
    print(max(weights))

    nx.draw_networkx_edges(H, positions, edgelist=edges,style='solid',
                          ###adds width=weights and edge_color = weights 
                          ###so that edges are based on the weight parameter 
                          ###edge_cmap is for the color scale based on the weight
                          ### edge_vmin and edge_vmax assign the min and max weights for the width
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                           edge_vmin = 0,#min(weights),#0.55,#min(weights), 
                           edge_vmax= 0.7,#max(weights),#0.6,#max(weights)
                           #edge_vmin = min(weights),#0.55,#min(weights), 
                           #edge_vmax= max(weights),#0.6,#max(weights)
                           )

    # displays the graph without axis
    plt.axis('off')
    #plt.legend(['r','r'])
    #saves image
    #plt.savefig("part5" + corr_direction + ".png", format="PNG")
    #plt.show() 