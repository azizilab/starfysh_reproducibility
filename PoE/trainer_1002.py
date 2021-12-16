from tqdm import tqdm
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
import torch.nn.functional as F

def final_loss(bce_loss_exp,
               bce_loss_sig,
               bce_loss_img,
               #bce_loss_poe, 
               bce_loss_poe_rna,
               bce_loss_poe_img, 
               mu_exp, logvar_exp,
               mu_sig, logvar_sig,
               mu_img, logvar_img, 
               mu_poe, logvar_poe,
               
               gsva_scores_binary_signature
              ):
    
    alpha = 5
    beta = 0.001

    # joint loss
    #Loss_IBJ = 2* (bce_loss_poe_rna+bce_loss_poe_img) + beta * (-0.5 * torch.sum(1+logvar_poe-mu_poe.pow(2)-logvar_poe.exp()))
    Loss_IBJ = 2* (bce_loss_poe_rna) + beta * (-0.5 * torch.sum(1+logvar_poe-mu_poe.pow(2)-logvar_poe.exp()))
 
    # multiple loss
    #Loss_IBM = (bce_loss_exp + #bce_loss_img + 
    #           beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp())) + 
    #           beta * (-0.5 * torch.sum(1+logvar_sig-mu_sig.pow(2)-logvar_sig.exp())) +
    #           beta * (-0.5 * torch.sum(1+logvar_img-mu_img.pow(2)-logvar_img.exp())) )
    Loss_IBM = (10*bce_loss_exp+bce_loss_img+
                beta * (-0.5 * torch.sum(1+logvar_exp-mu_exp.pow(2)-logvar_exp.exp()))+
                beta * (-0.5 * torch.sum(1+logvar_sig-mu_sig.pow(2)-logvar_sig.exp()))
               )
    
    Loss_sig = 1e4*F.binary_cross_entropy_with_logits(mu_sig, gsva_scores_binary_signature)
    #print('Loss_IBJ=', Loss_IBJ)
    #print('Loss_IBM=', Loss_IBM)
    #print('Loss_sig=', Loss_sig)

    return Loss_IBJ + alpha * Loss_IBM #+ Loss_sig




def train(model, dataloader, dataset, device, optimizer, criterion, criterion_img, sig_mask_df):
    model.train()

    running_loss = 0.0
    counter = 0
    for i,(data_exp,data_img,data_loc) in tqdm(enumerate(dataloader),total = int(len(dataset)/dataloader.batch_size)):
        
        counter +=1
        mini_batch,_,_= data_img.shape
        data_img = data_img.reshape(mini_batch,-1).float()
        optimizer.zero_grad()

        # gene expression, 1D data
        data_exp = data_exp.to(device)  
        recon_exp, recon_sig, mu_exp, logvar_exp, mu_sig, logvar_sig = model.predict_linVAE(data_exp, x_sample.to(device))

        # image, 2D data
        data_img = data_img.to(device)  
        recon_img, mu_img, logvar_img = model.predict_imgVAE(data_img)

        # POE
        recon_poe_rna, recon_poe_img, mu_poe,logvar_poe = model.predictor_POE(mu_exp, logvar_exp, mu_sig, logvar_sig, mu_img, logvar_img)

        # calculate loss
        bce_loss_exp = criterion(recon_exp, data_exp)
        bce_loss_img = criterion(recon_img, data_img)
        #print(recon_sig.shape)
        #print(x_sample.shape)
        bce_loss_sig = criterion(recon_sig, x_sample.to(device))
        bce_loss_poe_rna = criterion(recon_poe_rna,data_exp)
        bce_loss_poe_img = criterion(recon_poe_img,data_img)
        
        
        loss = final_loss(bce_loss_exp,
                         bce_loss_sig,
                         bce_loss_img, 
                         bce_loss_poe_rna,
                         bce_loss_poe_img, 
                         mu_exp, logvar_exp,
                         mu_sig, logvar_sig,
                         mu_img, logvar_img,
                         mu_poe, logvar_poe,
                         gsva_scores_binary_signature
                        ) 
        
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss