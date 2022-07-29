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
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


from torch.distributions import Normal, LogNormal,Dirichlet, kl_divergence as kl

class AVAE(nn.Module):
    """ model design ***
    p(x|z)=f(z)
    p(z|x)~N(0,1)
    q(z|x)~g(x)
    """
    
    def __init__(self,
                 adata,
                 gene_sig,
                 alpha_min,
                 win_loglib,
    ) -> None:
        """
                 
        """
        super(AVAE, self).__init__()
        self.win_loglib=torch.Tensor(win_loglib)
        
        self.c_in = adata.shape[1] # c_in : Num. input features (# input genes)
        self.c_bn = 10 # c_bn : latent number, numbers of bottle neck
        self.c_hidden = 256
        self.c_kn = gene_sig.shape[1]
        
        self.alpha = torch.nn.Parameter(torch.rand(self.c_kn)*1e3,requires_grad=True)

        
        self.c_enc = nn.Sequential(
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU()
        )
        
        self.c_enc_m = nn.Sequential(
                                nn.Linear(self.c_hidden, self.c_kn, bias=True),
                                nn.BatchNorm1d(self.c_kn, momentum=0.01,eps=0.001),
                                nn.Softmax(dim=-1)
        )
        #self.c_enc_logv = nn.Linear(self.c_hidden, self.c_hidden)
        
        self.l_enc = nn.Sequential(
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU(),
                                #nn.Linear(self.c_hidden, 1, bias=True),
                                #nn.ReLU(),
        )
        
        self.l_enc_m = nn.Linear(self.c_hidden, 1)
        self.l_enc_logv = nn.Linear(self.c_hidden, 1)
        
        # neural network f1 to get the z, p(z|x), f1(x,\phi_1)=[z_m,torch.exp(z_logv)]
        
        self.z_enc = nn.Sequential(
                                #nn.Linear(self.c_in+self.c_kn, self.c_hidden, bias=True),
                                nn.Linear(self.c_in, self.c_hidden, bias=True),
                                nn.BatchNorm1d(self.c_hidden, momentum=0.01,eps=0.001),
                                nn.ReLU(),
        )
        
        self.z_enc_m = nn.Linear(self.c_hidden, self.c_bn *  self.c_kn)
        self.z_enc_logv = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        
        # gene dispersion
        self.px_r = torch.nn.Parameter(torch.randn(self.c_in),requires_grad=True)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.px_hidden_decoder = nn.Sequential(
                                nn.Linear(self.c_bn, self.c_hidden, bias=True),
                                nn.ReLU(),         
        )
        self.px_scale_decoder = nn.Sequential(
                              nn.Linear(self.c_hidden,self.c_in),
                              nn.Softplus(),
                              #nn.Softmax(dim=-1),
                              #nn.Softmax(),
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
    
    def inference(self,
                  x,
                 ):
        #library = torch.log(x.sum(1)).unsqueeze(1)
        # l is inferred from logrithmized x
        
        x_n = torch.log(1+x)
        hidden = self.l_enc(x_n)
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)
        ql = torch.clamp(ql, min = 0.01)

        #ql = torch.exp(library)
        # x is processed by dividing the inferred library
        
        x_n = torch.log(1+x)
        #x_n = x / (ql+1)#(ql+1)
        #x_n = x_n / (x_n.sum(axis=1, keepdims=True)+1) #* self.alpha
        #x_n=torch.log(1+x) - ql
        hidden = self.c_enc(x_n)
        qc_m =  self.c_enc_m(hidden) 
        #print('qc_m',qc_m)
        #qc_logv = self.c_enc_logv(hidden)
        # sampling qc 
        #print('qc_p=',qc_p)
        

        qc = Dirichlet(self.alpha * qc_m).rsample()[:,:,None]
        hidden = self.z_enc(x_n)
        #hidden = self.z_enc(torch.concat([x_n,qc[:,:,0]],axis=1))
        #hidden = self.z_enc(qc)
        qz_m_ct = self.z_enc_m(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_m_ct = (qc * qz_m_ct)
        
        qz_m = qz_m_ct.sum(axis=1)
        qz_logv_ct = self.z_enc_logv(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_logv_ct = (qc * qz_logv_ct)
        
        qz_logv = qz_logv_ct.sum(axis=1)
        #qz_m = self.z_enc_m(hidden)
        #qz_logv = self.z_enc_logv(hidden)
        qz = self.reparameterize(qz_m, qz_logv)

        #print('qn=',qn)
        
        return dict(
                    qc_m = qc_m,
                    qc=qc,
                    qz_m=qz_m,
                    qz_m_ct=qz_m_ct,
                    qz_logv = qz_logv,
                    qz_logv_ct = qz_logv_ct,
                    qz=qz,
                    ql_m=ql_m,
                    ql_logv = ql_logv,
                    ql=ql
                   )
    
    def generative(self,
                   inference_outputs,
                   x,
                   xs_k,
                   library_i
                   
                  ):
        
        qz = inference_outputs['qz']
        ql = inference_outputs['ql']
        ql_m = inference_outputs['ql_m']
        
        hidden = self.px_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        px_rate = torch.exp(ql) * px_scale 
        
        
        #xs_k = xs_k/torch.exp(library_i) * torch.exp(library_i.mean(axis=1,keepdims=True))
        xs_k = xs_k / torch.exp(ql) * torch.exp(ql.mean(axis=1,keepdims=True))#* torch.exp(ql.min(axis=1,keepdims=True))
        #xs_k = torch.log(xs_k+1e-5)
        #xs_k = xs_k / (xs_k.sum(axis=1,keepdims=True)+1e-5)
        #xs_k = xs_k-xs_k.min()
        #xs_k = (xs_k-xs_k.min())/(xs_k.max()-xs_k.min()) 
        #xs_k = (xs_k-xs_k.min())/(xs_k.max()-xs_k.min())
        pc_p = self.alpha * xs_k + 1e-5#(np.log(xs_k+1)+1e-5)
        
        #xs_k = torch.log((xs_k/(ql))+1)+1e-5
        #xs_k = torch.log(xs_k/ql)+1e-5
        #xs_k  = xs_k/ xs_k.max()
        #xs_k  = xs_k/ xs_k.sum(axis=1, keepdims=True)
        

        return dict(
                    px_rate=px_rate,
                    px_r=self.px_r,
                    pc_p=pc_p,
                    xs_k=xs_k,

                   )
    
    
    def get_loss(self,
            generative_outputs,
            inference_outputs,
            x, 
            x_peri,
            library,
            device 
        ):    
    
        qc_m = inference_outputs["qc_m"]
        qc = inference_outputs["qc"]
        
        qz_m = inference_outputs["qz_m"]
        qz_logv = inference_outputs["qz_logv"]
        qz = inference_outputs["qz"]
        
        ql_m = inference_outputs["ql_m"]
        ql_logv = inference_outputs['ql_logv']
        ql = inference_outputs['ql']
        
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        pc_p = generative_outputs["pc_p"]
        

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_logv)
        
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(torch.exp(qz_logv))), Normal(mean, scale)).sum(
            dim=1
        ).mean()
     
        kl_divergence_n = kl(Normal(ql_m, torch.sqrt(torch.exp(ql_logv))), Normal(library,torch.ones_like(ql))).sum(
            dim=1
        ).mean()
        
        
        if (x_peri[:,0]==1).sum()>0:
            kl_divergence_c = kl(Dirichlet(qc_m[x_peri[:,0]==1]*self.alpha), Dirichlet(pc_p[x_peri[:,0]==1])).mean()
            if (x_peri[:,0]==0).sum()>0:
                #kl_divergence_c = kl_divergence_c+(self.win_loglib.max()-self.win_loglib.min())*1e-2*kl(Dirichlet(qc_m[x_peri[:,0]==0]*self.alpha), Dirichlet(pc_p[x_peri[:,0]==0])).mean()
                # para-dependent 
                if ((x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))).sum()>0:
                    kl_divergence_c = kl_divergence_c +  1e-1*kl(Dirichlet(qc_m[(x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))]*self.alpha), Dirichlet(pc_p[(x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))])).mean()
                #    #kl(Dirichlet(qc_m[x_peri[:,0]==0]*self.alpha2 * (ql[x_peri[:,0]==0]+1) ), Dirichlet(pc_p[x_peri[:,0]==0])).mean()
                if ((x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))).sum()>0:
                    kl_divergence_c = kl_divergence_c +  1e-2*kl(Dirichlet(qc_m[(x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))]*self.alpha), Dirichlet(pc_p[(x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))])).mean()
                #kl(Dirichlet(qc_m[x_peri[:,0]==0]*self.alpha2 * (ql[x_peri[:,0]==0]+1) ), Dirichlet(pc_p[x_peri[:,0]==0])).mean()
        else:
            kl_divergence_c = torch.Tensor([0.0])

        
        reconst_loss = -NegBinom(px_rate, torch.exp(px_r)).log_prob(x).sum(-1).mean()
        
        reconst_loss = reconst_loss.to(device)
        kl_divergence_z = kl_divergence_z.to(device)
        kl_divergence_c = kl_divergence_c.to(device)
        kl_divergence_n = kl_divergence_n.to(device)
        loss = reconst_loss + kl_divergence_z+ kl_divergence_c + kl_divergence_n

        return (loss,
                reconst_loss,
                kl_divergence_z,
                kl_divergence_c,
                kl_divergence_n
               )

    
def valid_model(model):
    
    model.eval()

    x_valid = torch.Tensor(np.array(adata_sample_filter.to_df()))
    x_valid = x_valid.to(device)
    gene_sig_exp_valid = torch.Tensor(np.array(gene_sig_exp_m)).to(device)
    library = torch.log(x_valid.sum(1)).unsqueeze(1)

    inference_outputs =  model.inference(x_valid)
    generative_outputs = model.generative(inference_outputs,library, gene_sig_exp_valid)

    qz_m = inference_outputs["qz_m"].detach().numpy()
    qc_m = inference_outputs["qc_m"].detach().numpy()
    qc = inference_outputs["qc"].detach().numpy()
    qz_logv = inference_outputs["qz_logv"].detach().numpy()
    qz = inference_outputs["qz"].detach().numpy()
    px_r = generative_outputs["px_r"].detach().numpy()
    pc_p = generative_outputs["pc_p"].detach().numpy()
    px_rate = generative_outputs["px_rate"].detach().numpy()
    ql = inference_outputs["ql"].detach().numpy()
    ql_m = inference_outputs["ql_m"].detach().numpy()
    px = NegBinom(generative_outputs["px_rate"], torch.exp(generative_outputs["px_r"])).sample().detach().numpy()
    

    corr_map_qcm = np.zeros([3,3])
    #corr_map_genesig = np.zeros([3,3])
    #for i in range(3):
    #    for j in range(3):
    #        corr_map_qcm[i,j], _ = pearsonr(qc_m[:,i], proportions.iloc[:,j])
            #corr_map_genesig[i,j], _ = pearsonr(gene_sig_exp_m.iloc[:,i], proportions.iloc[:,j])
    

    return 1/3*(corr_map_qcm[0,0]+corr_map_qcm[1,1]+corr_map_qcm[2,2])

from scipy.stats import pearsonr
from tqdm import tqdm
from torch.distributions import Normal, kl_divergence as kl
def train(
            model,
            dataloader,
            dataset,
            device,
            optimizer,
            
        ):
    model.train()
    
    running_loss = 0.0
    running_z = 0.0
    running_c = 0.0
    running_n = 0.0
    running_reconst = 0.0
    counter = 0
    corr_list = []
    for i, (x, xs_k, x_peri, library_i) in enumerate(dataloader):
        
        counter +=1
        x = x.float()
        x = x.to(device)
        xs_k = xs_k.to(device)
        x_peri = x_peri.to(device)
        library_i = library_i.to(device)
       

        #gene_sig_m = gene_sig_m+1e-5
        #gene_sig_m= gene_sig_m/gene_sig_m.sum(dim=1,keepdim=True)
        
        inference_outputs =  model.inference(x)
        #print('inference_outputs.shape',inference_outputs.shape)
        
        generative_outputs = model.generative(inference_outputs, x, xs_k,library_i)
        
        (loss,
         reconst_loss,
         kl_divergence_z,
         kl_divergence_c,
         kl_divergence_n
        ) = model.get_loss(generative_outputs,
                           inference_outputs,                                           
                           x,
                           x_peri,
                           library_i,
                           device
                       )
        
        optimizer.zero_grad()
        loss.backward()
        
        running_loss += loss.item()
        running_reconst +=reconst_loss.item()
        running_z +=kl_divergence_z.item()
        running_c +=kl_divergence_c.item()
        running_n +=kl_divergence_n.item()    
        
        optimizer.step()
        
        #corr_list.append(valid_model(model))
        
        
    train_loss = running_loss / counter
    train_reconst = running_reconst / counter
    train_z = running_z / counter
    train_c = running_c / counter
    train_n = running_n / counter
    
    
    
    
    return train_loss, train_reconst, train_z, train_c, train_n, corr_list




# Reference:
# https://github.com/YosefLab/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py
from torch.distributions import kl_divergence, Distribution
from torch.distributions import Gamma,Poisson
class NegBinom(Distribution):
    """
    Gamma-Poisson mixture approximation of Negative Binomial(mean, dispersion)

    lambda ~ Gamma(mu, theta)
    x ~ Poisson(lambda)
    """

    def __init__(self, mu, theta, eps=1e-5):
        """
        Parameters
        ----------
        mu : torch.Tensor
            mean of NegBinom. distribution
            shape - [# genes,]

        theta : torch.Tensor
            dispersion of NegBinom. distribution
            shape - [# genes,]
        """
        super(NegBinom, self).__init__(validate_args=False)
        assert (mu > 0).sum() and (theta > 0).sum(), \
            "Negative mean / dispersion of Negative detected"

        self.mu = mu
        self.theta = theta
        self.eps = eps

    def sample(self):
        #print('self.theta=',self.theta)
        #print('self.mu=',self.mu)
        #assert (self.theta +self.eps> 0).sum() and (self.mu +self.eps> 0)
        lambdas = Gamma(
            concentration=self.theta,
            rate=(self.theta+self.eps) / (self.mu+self.eps),
        ).rsample()

        x = Poisson(lambdas).sample()

        return x

    def log_prob(self, x):
        """log-likelihood"""
        ll = torch.lgamma(x + self.theta) - \
             torch.lgamma(x + 1) - \
             torch.lgamma(self.theta) + \
             self.theta * (torch.log(self.theta + self.eps) - torch.log(self.theta + self.mu + self.eps)) + \
             x * (torch.log(self.mu + self.eps) - torch.log(self.theta + self.mu + self.eps))

        return ll
    
    
def model_eval(model,adata_sample, sig_mean, device,library_i,lib_low):
    
    model.eval()
    library_i = torch.Tensor(library_i[:,None])
    x_valid = torch.Tensor(np.array(adata_sample.to_df()))
    x_valid = x_valid.to(device)
    gene_sig_exp_valid = torch.Tensor(np.array(sig_mean)).to(device)
    library = torch.log(x_valid.sum(1)).unsqueeze(1)

    inference_outputs =  model.inference(x_valid)
    generative_outputs = model.generative(inference_outputs,library, gene_sig_exp_valid, library_i)

    px = NegBinom(generative_outputs["px_rate"], torch.exp(generative_outputs["px_r"])).sample().detach().cpu().numpy()
    return inference_outputs, generative_outputs, px


def model_ct_exp(model,adata_sample, sig_mean, device,library_i,lib_low,ct_idx):

    model.eval()
    library_i = torch.Tensor(library_i[:,None])
    x_valid = torch.Tensor(np.array(adata_sample.to_df()))
    x_valid = x_valid.to(device)
    gene_sig_exp_valid = torch.Tensor(np.array(sig_mean)).to(device)
    library = torch.log(x_valid.sum(1)).unsqueeze(1)

    inference_outputs =  model.inference(x_valid)
    inference_outputs['qz'] = inference_outputs['qz_m_ct'][:,ct_idx,:]
    generative_outputs = model.generative(inference_outputs,library, gene_sig_exp_valid, library_i)

    px = NegBinom(generative_outputs["px_rate"], torch.exp(generative_outputs["px_r"])).sample().detach().cpu().numpy()
    return px
