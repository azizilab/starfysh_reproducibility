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

import anndata

import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from sklearn.neighbors import NearestNeighbors
import json


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
  