
from tqdm import tqdm
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)
import torch.nn.functional as F



def get_recon(trainset,patch_r,gene_sig,trainloader,model):

    recon_exp_stack = np.zeros([trainset.genexp.shape[0],trainset.genexp.shape[1]+trainset.genexp_sig.shape[1]])
    recon_img_stack = np.zeros([trainset.genexp.shape[0],patch_r*2,patch_r*2,1])
    recon_poe_img_stack = np.zeros([trainset.genexp.shape[0],patch_r*2,patch_r*2,1])
    recon_poe_rna_stack = np.zeros([trainset.genexp.shape[0],trainset.genexp.shape[1]+trainset.genexp_sig.shape[1]])

    wholeimg_stitch = np.ones([2000,2000,1])*(-1)
    recon_wholeimg_stitch = np.ones([2000,2000,1])*(-1)
    recon_poe_wholeimg_stitch = np.ones([2000,2000,1])*(-1)
    
    

    mu_exp_stack = np.zeros([trainset.genexp.shape[0],gene_sig.shape[1]])
    mu_img_stack = np.zeros([trainset.genexp.shape[0],gene_sig.shape[1]])
    mu_poe_stack = np.zeros([trainset.genexp.shape[0],gene_sig.shape[1]])

    for i,(adata_variable, adata_sig, adata_img,data_loc) in enumerate(trainloader):

        mini_batch , num_varibale_gene  = adata_variable.shape
        _ , num_sig_gene = adata_sig.shape
        
        adata_img = adata_img.reshape(mini_batch,-1).float() # flatten the img
        

        # gene expression, 1D data
        adata_variable = adata_variable.to(device)  
        adata_sig = torch.Tensor(adata_sig).to(device)  
        recon_exp, recon_peri, mu_exp, logvar_exp, mu_peri, logvar_peri = model.predict_linVAE(adata_variable , adata_sig,  x_sample_variable.to(device),  x_sample_sig.to(device) )
        
        # image, 2D data
        adata_img = adata_img.to(device)  
        recon_img, mu_img, logvar_img = model.predict_imgVAE(adata_img)
        
        # POE
        recon_poe_rna, recon_poe_img, mu_poe,logvar_poe = model.predictor_POE(mu_exp, logvar_exp, mu_peri, logvar_peri, mu_img, logvar_img, adata_sig, x_sample_variable.to(device),  x_sample_sig.to(device) )
        
        for ii in range(mini_batch):
            loc = np.where(data_loc[ii]==map_info.index)
            recon_exp_stack[loc[0][0]] = recon_exp[ii].cpu().detach().numpy()
            recon_img_stack[loc[0][0]] = recon_img.reshape([recon_img.shape[0],patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy()
            recon_poe_img_stack[loc[0][0]] = recon_poe_img.reshape([recon_poe_img.shape[0],patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy()
            recon_poe_rna_stack[loc[0][0]] = recon_poe_rna[ii].cpu().detach().numpy()
          
            mu_exp_stack[loc[0][0]] = mu_exp[ii].cpu().detach().numpy()
            mu_img_stack[loc[0][0]] = mu_img[ii].cpu().detach().numpy()
            mu_poe_stack[loc[0][0]] = mu_poe[ii].cpu().detach().numpy()

            image_col = map_info.iloc[loc[0][0]]['imagecol']
            image_row = map_info.iloc[loc[0][0]]['imagerow']
        
            image_slide_z =0
        
            recon_wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = recon_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
            wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = adata_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
            recon_poe_wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = recon_poe_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
            
            adata_df_concat = pd.concat([adata_df_variable,adata_df_signature], axis=1)
            
            recon_results = {'recon_exp_stack':recon_exp_stack,
                             'recon_img_stack':recon_img_stack,
                             'recon_poe_img_stack':recon_poe_img_stack,
                             'recon_poe_rna_stack':recon_poe_rna_stack,
                             'wholeimg_stitch':wholeimg_stitch,
                             'recon_wholeimg_stitch':recon_wholeimg_stitch,
                             'recon_poe_wholeimg_stitch':recon_poe_wholeimg_stitch,
                             'mu_exp_stack':mu_exp_stack,
                             'mu_img_stack':mu_img_stack,
                             'mu_poe_stack':mu_poe_stack,
                             'model':model,
                             'train_loss':train_loss,
                             'adata_df_concat':adata_df_concat,
                             'map_info':map_info,
                             'mean_arch_ct':mean_arch_ct
                            }
            
    return recon_results

        