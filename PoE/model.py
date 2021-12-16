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



class poe_model_sig(nn.Module):

    def __init__(self):

        super(poe_model_sig, self).__init__()

        self.c_bn = len(sig_vars) # latent number, numbers of bottle neck
        self.c_in = len(gene_final) # number of genes 
        self.sig_vars_prior = torch.Tensor(sig_vars).to(device) # variance of signature/pure spots
        
        # For gene expression
        self.linVAE_enc = nn.Sequential(nn.Linear(self.c_in, 128, bias=True),
                                  nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                                  nn.ReLU(),
                                  )
        self.linVAE_fc_mu = nn.Linear(128, self.c_bn)
        self.linVAE_fc_logvar = nn.Linear(128, self.c_bn)

        self.linVAE_z_fc = nn.Linear(self.c_bn, 128)
        self.linVAE_dec = nn.Sequential(
                                  nn.Linear(128, self.c_in,bias=True),
                                  nn.BatchNorm1d(self.c_in, momentum=0.01, eps=0.001),
                                  nn.ReLU(),
                                  )
        
        # For image 
        self.imgVAE_enc = nn.Sequential(nn.Linear(patch_r*patch_r*4, 128, bias=True), # flatten the images into 1D
                                  nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                                  nn.ReLU(),                                 
                                  )
        self.imgVAE_fc_mu = nn.Linear(128, self.c_bn)
        self.imgVAE_fc_logvar = nn.Linear(128, self.c_bn)

        self.imgVAE_z_fc = nn.Linear(self.c_bn, 128)
        self.imgVAE_dec = nn.Sequential(nn.Linear(128, patch_r*patch_r*4, bias=True),
                                  nn.BatchNorm1d(patch_r*patch_r*4, momentum=0.01, eps=0.001),
                                  nn.ReLU(),                                 
                                  )
        
        self.POE_z_fc = nn.Linear(self.c_bn, 128)
        self.POE_dec = nn.Sequential(nn.Linear(128, 512, bias=True),
                                     nn.BatchNorm1d(512, momentum=0.01, eps=0.001),
                                     nn.ReLU(), 
                                     nn.Linear(512, 1024, bias=True),
                                     nn.BatchNorm1d(1024, momentum=0.01, eps=0.001),
                                     nn.ReLU(), 
                                     nn.Linear(1024, self.c_in,bias=True),
                                     nn.BatchNorm1d(self.c_in, momentum=0.01, eps=0.001),
                                     nn.ReLU(),
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


    def predict_linVAE(self, x, x_sig):

        batch, _, = x.shape
        hidden = self.linVAE_enc(x)
        hidden_sig = self.linVAE_enc(x_sig)
        
        mu = self.linVAE_fc_mu(hidden)
        mu_sig = self.linVAE_fc_mu(hidden_sig)
        #log_var = self.linVAE_fc_logvar(hidden)
        #use the prior std instead of fc
        sig_stds = torch.sqrt(self.sig_vars_prior)
        log_var = sig_stds.repeat(mu.shape[0], 1)
        log_var_sig = sig_stds.repeat(mu_sig.shape[0], 1)

        z = self.reparameterize(mu, log_var)
        z_sig = self.reparameterize(mu_sig, log_var_sig)

        x = self.linVAE_z_fc(z)
        x_sig = self.linVAE_z_fc(z_sig)
        recon = self.linVAE_dec(x)
        recon_sig = self.linVAE_dec(x_sig)

        return recon, recon_sig, mu, log_var, mu_sig, log_var_sig
    
    def predict_imgVAE(self, x):

        batch, _= x.shape
        hidden = self.imgVAE_enc(x)
    
        mu = self.imgVAE_fc_mu(hidden)  
        log_var = self.imgVAE_fc_logvar(hidden)

        z = self.reparameterize(mu, log_var)

        x = self.imgVAE_z_fc(z) 
        reconstruction = self.imgVAE_dec(x)

        return reconstruction, mu, log_var

    def predictor_POE(self, mu_exp, logvar_exp, mu_sig, logvar_sig, mu_img, logvar_img):

        var_poe = torch.div(1., 
                            1 + 
                            torch.div(1., torch.exp(logvar_exp)) + 
                            #torch.div(1., torch.exp(logvar_sig)) + 
                            torch.div(1., torch.exp(logvar_img))
                            )
        
        mu_poe = var_poe * (0 + 
                            mu_exp * torch.div(1., torch.exp(logvar_exp)+1e-5) + 
                            #mu_sig * torch.div(1., torch.exp(logvar_sig)+1e-5) +
                            mu_img * torch.div(1., torch.exp(logvar_img)+1e-5)                           
                            )
        
        z = self.reparameterize(mu_poe, torch.log(var_poe+0.001))
        
        x = self.POE_z_fc(z)

        batch, _, = x.shape
        reconstruction =  self.POE_dec(x)

        return reconstruction, mu_poe,torch.log(var_poe+0.001)
    