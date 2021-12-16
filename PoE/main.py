## prepare image
from __future__ import print_function

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
import torch
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


# import PoE module
import PoE.utils as utils
import PoE.dataloader_1002 as dataloader
import PoE.utils_plots as utils_plots

import PoE.dataloader as  dataloader
import anndata

import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#%matplotlib inline
import cv2
#plt.rcParams['figure.figsize'] = 15, 15
#plt.rcParams['image.cmap'] = 'gray'
#titlesize = 24
from sklearn.neighbors import NearestNeighbors
import json



# deadling with gsva and siganture
def discretize_gsva(gsva_raw, thld=0.7):
    """
    Discretize GSVA scores for training
    
    Parameters
    ----------
    gsva_raw : pd.DataFrame
        raw GSVA matrix (shape: [S x C])
    
    """
    gsva_train = gsva_raw.apply(
        lambda x: x >= x.quantile(thld), 
        axis=0
    ).astype(np.uint8)
    
    return gsva_train

def calc_var_priors(adata, sig_genes_dict):
    """Calculate avg. gexp variance for each signature as priors"""
    # Subset signature genes as Union(marker_genes, highly_variable genes)
    marker_genes = list(set(
        [gene
         for genes in sig_genes_dict.values() 
         for gene in genes]
    ))
    hv_genes = adata.var_names[adata.var['highly_variable']]
    sel_genes = set(np.union1d(marker_genes, hv_genes))
    genes_dict = {}
    for ct, genes in sig_genes_dict.items():
        filtered_genes = [gene for gene in genes if gene in sel_genes]
        if len(filtered_genes) > 0:
            genes_dict[ct] = filtered_genes
            
    # Calculate signature variance priors
    sig_vars = []
    sig_vars_dict = {} 
    
    for ct, genes in genes_dict.items():
        spot_mask = adata.obs['perif_val'].apply(
            lambda x: ct in x
        ).astype(bool)
        gene_mask = adata.var_names.intersection(genes)
        ct_vars = adata[spot_mask, gene_mask].X.sum(axis=1).var()
        sig_vars.append(ct_vars)
        sig_vars_dict[ct] = ct_vars
        
    sig_vars = np.asarray(sig_vars)
    
    return sig_vars, sig_vars_dict, sel_genes

def find_spots(adata, gsva, n_nbrs=20):
    """
    Find spots representing `pure spots` & their nearest neighbors
    based on GSVA scores for each cell type
    """
    assert 'highly_variable' in adata.var_keys(), \
        "Please choose highly variable genes first"
    
    # Calculate distance with only highly variable genes
    embedding = adata[:, adata.var['highly_variable']].X.A
    pure_idxs = np.argmax(gsva.values, axis=0)
    pure_spots = gsva.idxmax(axis=0)
    
    pure_dict = {
        spot: ct 
        for (spot, ct) in zip(pure_spots, gsva.columns)
    }
    
    nbrs = NearestNeighbors(n_neighbors=n_nbrs+len(pure_spots)).fit(embedding)
    nn_graph = nbrs.kneighbors(embedding)[1] 
    
    perif_spots = []        
    perif_idxs = [nn_graph[idx].tolist() for idx in pure_idxs]
    for i, raw_idxs in enumerate(perif_idxs):
        idxs = [idx 
                for idx in raw_idxs 
                if idx not in pure_idxs or idx == raw_idxs[0]]
        perif_spots.append(gsva.index[idxs[:n_nbrs]])
    
    perif_dict = {}
    for (spots, ct) in zip(perif_spots, gsva.columns):
        for spot in spots:
            if spot not in perif_dict.keys():
                perif_dict[spot] = [ct]
            else:
                perif_dict[spot].append(ct)
        
    pure_spots = np.asarray(pure_spots)
    perif_spots = np.asarray(perif_spots).flatten()
    
    adata.obs['pure_val'] = [
        pure_dict[spot] 
        if spot in pure_spots else 'nan' for spot in adata.obs_names 
    ]
    adata.obs['perif_val'] = [
        perif_dict[spot]
        if spot in perif_spots else [np.nan] for spot in adata.obs_names 
    ]
    adata.obs['perif_unique_val'] = adata.obs['perif_val'].apply(lambda x: x[0])

    return pure_spots, perif_spots
    

#define the varibale geneset, and signature geneset, and create dataloader
def gene_for_train(adata, df_sig):
    """find the varibale gene name, the mattered gene signatures, and the combined variable+signature for training"""
    
    variable_gene = adata.var_names[adata.var['highly_variable']]
    print('the number of original variable genes in the dataset',variable_gene.shape)
    print('the number of siganture genes in the dataset',np.unique(df_sig.values.flatten().astype(str)).shape)
    
    # filter out some genes in the signature not in the var_names
    sig_gname_filtered =  np.intersect1d(adata.var_names,np.unique(df_sig.values.flatten().astype(str)))
    print('after filter out some genes in the signature not in the var_names ...',sig_gname_filtered.shape)
    # filter out some genes not highly expressed in the signature
    sig_gname_filtered = sig_gname_filtered[adata.to_df().loc[:,sig_gname_filtered].sum()>0]
    print('after filter out some genes not highly expressed in the signature ...', sig_gname_filtered.shape)
    
    sig_variable_gene_inter  = set([*np.array(variable_gene) ,*sig_gname_filtered])
    print('combine the varibale and siganture, the total unique gene number is ...', len(sig_variable_gene_inter))
    
    return variable_gene, sig_gname_filtered, sig_variable_gene_inter
class poe_model_sig(nn.Module):

    def __init__(self, adata_df_variable, adata_df_signature, patch_r, sig_vars,device):
        
        super(poe_model_sig, self).__init__()
        self.c_bn = len(sig_vars) # latent number, numbers of bottle neck
        self.c_in = adata_df_variable.shape[1]+adata_df_signature.shape[1] # number of genes
        self.patch_r = patch_r
        
        self.sig_vars_prior = torch.Tensor(sig_vars).to(device) # variance of signature/pure spots
        
        # For gene expression
        self.linVAE_enc = nn.Sequential(nn.Linear(self.c_in, 128, bias=True),
                                  nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                                  nn.ReLU(),
                                  )
        self.linVAE_fc_mu = nn.Linear(128+adata_df_signature.shape[1], self.c_bn)
        
        self.linVAE_z_fc = nn.Linear(self.c_bn, 128)
        
        self.linVAE_dec = nn.Sequential(
                                  nn.Linear(128+adata_df_signature.shape[1], self.c_in,bias=True),
                                  nn.BatchNorm1d(self.c_in, momentum=0.01, eps=0.001),
                                  #nn.ReLU(),
                                  )
        # For image 
        self.imgVAE_enc = nn.Sequential(nn.Linear(self.patch_r*self.patch_r*4, 128, bias=True), # flatten the images into 1D
                                  nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                                  nn.ReLU(),                                 
                                  )
        self.imgVAE_fc_mu = nn.Linear(128, self.c_bn)
        self.imgVAE_fc_logvar = nn.Linear(128, self.c_bn)
        

        self.imgVAE_z_fc = nn.Linear(self.c_bn, 128)
        self.imgVAE_dec = nn.Sequential(nn.Linear(128, 256, bias=True),
                                  nn.BatchNorm1d(256, momentum=0.01, eps=0.001),
                                  nn.ReLU(),  
                                  nn.Linear(256, self.patch_r*self.patch_r*4, bias=True),
                                  nn.BatchNorm1d(self.patch_r*self.patch_r*4, momentum=0.01, eps=0.001),
                                  #nn.ReLU(),                                 
                                  )
        
        # PoE
        self.POE_z_fc = nn.Linear(self.c_bn, 128)
        
        self.POE_dec_rna = nn.Sequential(
                                     nn.Linear(128+adata_df_signature.shape[1], self.c_in, bias=True),
                                     nn.BatchNorm1d(self.c_in, momentum=0.01, eps=0.001),
                                     #nn.ReLU(), 
                                  )
        
        self.POE_dec_img = nn.Sequential(      
                                  nn.Linear(128, self.patch_r*self.patch_r*4, bias=True),
                                  nn.BatchNorm1d(self.patch_r*self.patch_r*4, momentum=0.01, eps=0.001),
                                  #nn.ReLU(),           
                                  )
        
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    
    def predict_linVAE(self, x, x_sig, x_peri, x_peri_sig):     
        
        x_concat = torch.cat((x,x_sig),1)
        x_peri_concate = torch.cat((x_peri,x_peri_sig),1)
        batch, _ = x_concat.shape
        
        hidden = self.linVAE_enc(x_concat)
        hidden_peri = self.linVAE_enc(x_peri_concate)
        
        mu = self.linVAE_fc_mu(torch.cat((hidden,x_sig),1))
        mu_peri = self.linVAE_fc_mu(torch.cat((hidden_peri,x_peri_sig),1))
        
        sig_stds = torch.sqrt(self.sig_vars_prior)
        
        log_var = sig_stds.repeat(mu.shape[0], 1)
        log_var_peri = sig_stds.repeat(mu_peri.shape[0], 1)
        
        z = self.reparameterize(mu, log_var)
        z_peri = self.reparameterize(mu_peri, log_var_peri)
        
        x = self.linVAE_z_fc(z)
        x_peri = self.linVAE_z_fc(z_peri)
        
        recon = self.linVAE_dec(torch.cat((x,x_sig),1))
        recon_peri = self.linVAE_dec(torch.cat((x_peri,x_peri_sig),1))
        
        return recon, recon_peri, mu, log_var, mu_peri, log_var_peri
    
        
            
    def predict_imgVAE(self, x):

        batch, _= x.shape
        hidden = self.imgVAE_enc(x)
    
        mu = self.imgVAE_fc_mu(hidden)  
        log_var = self.imgVAE_fc_logvar(hidden)

        z = self.reparameterize(mu, log_var)

        x = self.imgVAE_z_fc(z) 
        reconstruction = self.imgVAE_dec(x)

        return reconstruction, mu, log_var
    
    
    def predictor_POE(self, mu_exp, logvar_exp, mu_peri, logvar_peri, mu_img, logvar_img, adata_sig,  x_peri, x_peri_sig):
        
        batch, _ = mu_exp.shape
        var_poe = torch.div(1., 
                            1 + 
                            torch.div(1., torch.exp(logvar_exp)) + 
                            torch.div(1., torch.exp(logvar_img))
                            )
        
        mu_poe = var_poe * (0 + 
                            mu_exp * torch.div(1., torch.exp(logvar_exp)+1e-5) + 
                            mu_img * torch.div(1., torch.exp(logvar_img)+1e-5)                           
                            )
        
        z = self.reparameterize(mu_poe, torch.log(var_poe+0.001))
        
        x = self.POE_z_fc(z)
        
        reconstruction_rna =  self.POE_dec_rna(torch.cat((x,adata_sig),1))
        #reconstruction_rna_peri =  self.POE_dec_rna(torch.cat((x_peri,x_peri_sig),1))
        
        reconstruction_img =  self.POE_dec_img(x)
        
        return reconstruction_rna , reconstruction_img, mu_poe,torch.log(var_poe+0.001)
    
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

def final_loss(bce_loss_exp,
               bce_loss_peri,
               bce_loss_img, 
               bce_loss_poe_rna,
               bce_loss_poe_img, 
               
               mu_exp, logvar_exp,
               mu_peri, logvar_peri,
               mu_img, logvar_img,
               mu_poe, logvar_poe,
               gsva_sig_poe,
               device,
               gsva_sig
              ):
    alpha = 5
    beta = 0.001
    # joint loss
    Loss_IBJ = (bce_loss_poe_rna + bce_loss_poe_img) + beta * (-0.5 * torch.sum(1+logvar_poe-mu_poe.pow(2)-logvar_poe.exp()))
    
    # multiple loss
    Loss_IBM = (10*bce_loss_exp+50*bce_loss_img+
                beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
                beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
                beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
               )
    
    Loss_sig = 1e4*F.binary_cross_entropy_with_logits(mu_peri.to(device), gsva_sig.to(device))
    Loss_x = 1e4*F.binary_cross_entropy_with_logits(mu_exp.to(device), gsva_sig_poe.to(device))
    Loss_sig_poe = 1e4*F.binary_cross_entropy_with_logits(mu_poe.to(device), gsva_sig_poe.to(device))
    
    return Loss_IBJ + alpha * Loss_IBM + Loss_sig+ Loss_sig_poe + Loss_x
def train(model, dataloader, dataset, device, optimizer, criterion, criterion_img,x_sample_variable,x_sample_sig,gsva_scores_train,gsva_sig):
    model.train()
    
    running_loss = 0.0
    counter = 0
    
    for i,(adata_variable, adata_sig, adata_img,data_loc) in tqdm(enumerate(dataloader),total = int(len(dataset)/dataloader.batch_size)):
        
        #print(data_loc)
    
        counter +=1
        mini_batch , num_varibale_gene  = adata_variable.shape
        _ , num_sig_gene = adata_sig.shape
        
        adata_img = adata_img.reshape(mini_batch,-1).float() # flatten the img
        
        optimizer.zero_grad()
        
        
        # gene expression, 1D data
        adata_variable = adata_variable.to(device)  
        adata_sig = torch.Tensor(adata_sig).to(device)  
        recon_exp, recon_peri, mu_exp, logvar_exp, mu_peri, logvar_peri = model.predict_linVAE(adata_variable , adata_sig,  x_sample_variable.to(device),  x_sample_sig.to(device))
        
        # image, 2D data
        adata_img = adata_img.to(device)  
        recon_img, mu_img, logvar_img = model.predict_imgVAE(adata_img)
        
        # POE
        recon_poe_rna, recon_poe_img, mu_poe,logvar_poe = model.predictor_POE(mu_exp, logvar_exp, mu_peri, logvar_peri, mu_img, logvar_img, adata_sig,x_sample_variable.to(device),  x_sample_sig.to(device) )
        
        # calculate loss
        bce_loss_exp = criterion(recon_exp, torch.cat((adata_variable,adata_sig),1))
        bce_loss_peri = criterion(recon_peri, torch.cat((x_sample_variable.to(device),x_sample_sig.to(device)),1)) 
        
        bce_loss_img = criterion(recon_img, adata_img)
        
        bce_loss_poe_rna = criterion(recon_poe_rna,torch.cat((adata_variable,adata_sig),1))
        bce_loss_poe_img = criterion(recon_poe_img,adata_img)
        
        gsva_sig_poe = torch.Tensor(gsva_scores_train.loc[data_loc, :].to_numpy())
        
        loss = final_loss(bce_loss_exp,
                         bce_loss_peri,
                         bce_loss_img, 
                         bce_loss_poe_rna,
                         bce_loss_poe_img, 
                         
                         mu_exp, logvar_exp,
                         mu_peri, logvar_peri,
                         mu_img, logvar_img,
                         mu_poe, logvar_poe,
                         gsva_sig_poe,
                         device,
                         gsva_sig
                        
                        ) 
        #gsva_sig_all
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss


def run_main(data_path,adata_sample,sample_id):
    
    #sample_id = adata_sample.obs['sample']
    adata_sample = utils.exp_preprocess(adata_sample,n_top_genes=2000) 
    
    # load signature data [type x gene]
    signature_path = data_path
    sig_fname = os.path.join(signature_path,  'bc_signatures_Overall_sorted_1122.csv')
    df_sig = pd.read_csv(sig_fname)
    df_sig.columns = df_sig.columns.str.replace('.', ' ')
    df_sig.columns = df_sig.columns.str.replace('_', ' ')
    df_sig.columns = df_sig.columns.str.replace('-', ' ')
    df_sig.columns = df_sig.columns.str.replace('+', '')
    
    # prepare gsva
    
    df_gsva_raw= pd.read_csv(os.path.join(data_path, sample_id+'_gsva.csv'), index_col=0).transpose()
    df_gsva_raw.index = df_gsva_raw.index+'-'+sample_id
    df_gsva_raw.columns = df_sig.columns
    
    # image preparation

    adata_image = io.imread(os.path.join(data_path, sample_id, 'spatial','tissue_hires_image.png'))
    adata_image = (adata_image*255).astype(np.uint8)
    stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
    # specify stains of input image
    stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stains
    # create stain matrix
    # create stain matrix
    W = np.array([stain_color_map[st] for st in stains]).T
    # perform standard color deconvolution
    imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(adata_image, W)

    adata_image = imDeconvolved.Stains[:,:,0]
    adata_image = ((adata_image - adata_image.min())/(adata_image.max()-adata_image.min()) *255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))

    adata_image = clahe.apply(adata_image)

    adata_image = (adata_image - adata_image.min())/(adata_image.max()-adata_image.min())
    #adata_image = np.clip(adata_image, 0.2, 0.8)
    #adata_image = (adata_image - adata_image.min())/(adata_image.max()-adata_image.min())
    
    #Mapping images to location
    
    map_info = []


 

    f = open(os.path.join(data_path, sample_id, 'spatial','scalefactors_json.json',))
    json_info = json.load(f)
    f.close()
    tissue_hires_scalef = json_info['tissue_hires_scalef']
    
    tissue_position_list = pd.read_csv(os.path.join(data_path, sample_id, 'spatial','tissue_positions_list.csv'),header=None, index_col=0)
    tissue_position_list.index = tissue_position_list.index+'-'+sample_id
    tissue_position_list = tissue_position_list.loc[adata_sample.obs.index,:]
    map_info= tissue_position_list.iloc[:,1:3]
    map_info.columns=['array_row','array_col']
    map_info.loc[:,'imagerow'] = tissue_position_list.iloc[:,3] *tissue_hires_scalef
    map_info.loc[:,'imagecol'] = tissue_position_list.iloc[:,4] *tissue_hires_scalef
    map_info.loc[:,'sample'] = sample_id
    
    variable_gene, sig_gname_filtered, sig_variable_gene_inter = gene_for_train(adata_sample, df_sig)
    
    # prepare dataset for training
    adata_df_variable = adata_sample.to_df().loc[:,variable_gene]
    adata_df_signature = adata_sample.to_df().loc[:,sig_gname_filtered]
    
    # create data loader
    batch_size = 8
    patch_r = 13
    trainset = dataloader.shenet_traindatastack([adata_df_variable, adata_df_signature, adata_image],map_info,patch_r)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    # define the pure and peri spots

    pure_spots, perif_spots = find_spots(adata_sample, df_gsva_raw)
    spot_sizes = adata_sample.obs_names.map(lambda x: 500 if x in set(pure_spots) else 50)
    # Configure signature & gsva dataframes
    sig_genes_dict = {
        sig: df_sig[sig][~pd.isna(df_sig)[sig]].tolist() 
        for sig in df_sig.columns
    }

    # Discretize gsva scores given different thresholds
    gsva_scores_train = discretize_gsva(df_gsva_raw)
    
    
    sig_vars, sig_vars_dict, sel_genes = calc_var_priors(adata_sample, sig_genes_dict) 
    
    x_sample_variable = torch.Tensor(adata_df_variable.loc[perif_spots].to_numpy())
    x_sample_sig = torch.Tensor(adata_df_signature.loc[perif_spots].to_numpy())

    gsva_sig = torch.Tensor(gsva_scores_train.loc[perif_spots, :].to_numpy())
    
    #gsva_sig_all = torch.Tensor(gsva_scores_train.to_numpy())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #import PoE.model_1002 as poe_model
    # initialize the model
    model = poe_model_sig(adata_df_variable, adata_df_signature, patch_r,sig_vars, device).to(device)

    train_loss = []
    # set the learning parameters
    lr1 = 0.001
    #lr2 = 0.0001
    optimizer1 = optim.Adam(model.parameters(), lr=lr1)
    #optimizer2 = optim.Adam(model.parameters(), lr=lr2)
    criterion = nn.MSELoss(reduction='sum')
    criterion_img = nn.BCELoss(reduction='mean')
    
    epochs= 30
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(model,
                                 trainloader, 
                                 trainset, 
                                 device,
                                 optimizer1, 
                                 criterion,
                                 criterion_img, 
                                 x_sample_variable, 
                                 x_sample_sig,
                                 gsva_scores_train,
                                 gsva_sig
                                )
        train_loss.append(train_epoch_loss)
        torch.cuda.empty_cache()
        print(f"Train Loss: {train_epoch_loss:.4f}")
    print('finish lower training rate')
    
    torch.save(model.state_dict(), sample_id+'1215_model_single.pt')
    utils.save_loss_plot(train_loss, train_loss,sample_id)
    
    recon_exp_stack = np.zeros([trainset.genexp.shape[0],trainset.genexp.shape[1]+trainset.genexp_sig.shape[1]])
    recon_img_stack = np.zeros([trainset.genexp.shape[0],patch_r*2,patch_r*2,1])
    recon_poe_img_stack = np.zeros([trainset.genexp.shape[0],patch_r*2,patch_r*2,1])
    recon_poe_rna_stack = np.zeros([trainset.genexp.shape[0],trainset.genexp.shape[1]+trainset.genexp_sig.shape[1]])

    wholeimg_stitch = np.ones([2000,2000,1])*(-1)
    recon_wholeimg_stitch = np.ones([2000,2000,1])*(-1)
    recon_poe_wholeimg_stitch = np.ones([2000,2000,1])*(-1)

    mu_exp_stack = np.zeros([trainset.genexp.shape[0],27])
    mu_img_stack = np.zeros([trainset.genexp.shape[0],27])
    mu_poe_stack = np.zeros([trainset.genexp.shape[0],27])

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
        
          sample_id_ii = data_loc[ii].split('-')[-1]
          image_slide_z =np.where(np.array(sample_id)==sample_id_ii)[0][0]
        
          recon_wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = recon_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
          wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = adata_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
          recon_poe_wholeimg_stitch[int(image_row)-patch_r:int(image_row)+patch_r,int(image_col)-patch_r:int(image_col)+patch_r,image_slide_z] = recon_poe_img.reshape([mini_batch,patch_r*2,patch_r*2,1])[ii].cpu().detach().numpy().reshape([patch_r*2,patch_r*2])
            
          adata_df_concat = pd.concat([adata_df_variable,adata_df_signature], axis=1)
        
            
            
    return  recon_exp_stack, recon_img_stack, recon_poe_img_stack, recon_poe_rna_stack, wholeimg_stitch ,recon_wholeimg_stitch, recon_poe_wholeimg_stitch, mu_exp_stack, mu_img_stack, mu_poe_stack