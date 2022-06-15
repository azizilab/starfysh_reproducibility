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
               gsva_sig,
               mean_arch_ct,
               criterion
              ):
    alpha = 5
    beta = 0.001
    # joint loss
    Loss_IBJ = (bce_loss_poe_rna + bce_loss_poe_img) + beta * (-0.5 * torch.sum(1+logvar_poe-mu_poe.pow(2)-logvar_poe.exp()))
    #Loss_IBJ = (1e3*bce_loss_poe_rna + bce_loss_poe_img) + beta * (-0.5 * torch.sum(1+logvar_poe-mu_poe.pow(2)-logvar_poe.exp()))
    
    # multiple loss
    #Loss_IBM = (10*bce_loss_exp+50*bce_loss_img+
    #            beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
    #            beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
    #            beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
    #           )
    #Loss_IBM = (500*bce_loss_exp+5*bce_loss_img+
    #            beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
    #            beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
    #            beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
    #           )
    
    #Loss_IBM = (10*bce_loss_exp+50*bce_loss_img+
    #            beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
    #            beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
    #            beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
    #           )
    #Loss_IBM = (5e4*bce_loss_exp+1*bce_loss_img+
    #            5e2*beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
    #            5e2*beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
    #            beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
    #           )
    Loss_IBM = (5e5*bce_loss_exp+1*bce_loss_img+
                5e3*beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
                5e3*beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
                beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
               )
    #Loss_IBM = (bce_loss_exp+bce_loss_img+
    #            1*beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
    #            1*beta * (-0.5 * torch.sum(1+logvar_peri-mu_peri.pow(2)-logvar_peri.exp())) +
    #            beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) 
    #           )
    
    #Loss_sig = 1e4*F.binary_cross_entropy_with_logits(mu_peri.to(device), gsva_sig.to(device))
    #Loss_x = 1e4*F.binary_cross_entropy_with_logits(mu_exp.to(device), gsva_sig_poe.to(device))
    #Loss_anchor =  1e5*criterion(mu_peri, torch.Tensor(mean_arch_ct).to(device))
    #Loss_sig_poe = 1e4*F.binary_cross_entropy_with_logits(mu_poe.to(device), gsva_sig_poe.to(device))
    
    Loss_sig = 1e4*F.binary_cross_entropy_with_logits(mu_peri.to(device), gsva_sig.to(device))
    Loss_x = 1e4*F.binary_cross_entropy_with_logits(mu_exp.to(device), gsva_sig_poe.to(device))
    Loss_anchor =  1e5*criterion(mu_peri, torch.Tensor(mean_arch_ct).to(device))
    Loss_sig_poe = 1e6*F.binary_cross_entropy_with_logits(mu_poe.to(device), gsva_sig_poe.to(device))
    
    #Loss_sig = F.binary_cross_entropy_with_logits(mu_peri.to(device), gsva_sig.to(device))
    #Loss_x = F.binary_cross_entropy_with_logits(mu_exp.to(device), gsva_sig_poe.to(device))
    #Loss_anchor =  alpha*criterion(mu_peri, torch.Tensor(mean_arch_ct).to(device))
    #Loss_sig_poe = beta*F.binary_cross_entropy_with_logits(mu_poe.to(device), gsva_sig_poe.to(device))
    
    return Loss_anchor+Loss_IBJ + alpha * Loss_IBM + Loss_sig+ Loss_sig_poe + Loss_x



def train(model, dataloader, dataset, device, optimizer, criterion, criterion_img,x_sample_variable,x_sample_sig,gsva_scores_train,gsva_sig,mean_arch_ct):
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
                         gsva_sig,
                         mean_arch_ct,
                          criterion
                        
                        ) 
        #gsva_sig_all
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss
