from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import sys
import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim as optim
import torch.nn as nn

from skimage import io
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.distributions import Normal, LogNormal,Dirichlet, kl_divergence as kl

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class AVAE(nn.Module):
    """ model design ***
    p(x|z)=f(z)
    p(z|x)~N(0,1)
    q(z|x)~g(x)
    """
    
    def __init__(
        self,
        adata,
        gene_sig,
        win_loglib,
    ) -> None:
        """
        Auxiliary Variational AutoEncoder (AVAE) - Core model for
        spatial deconvolution without H&E image integration

        Paramters
        ---------
        adata : sc.AnnData
            ST raw expression count (dim: [S, G])

        gene_sig : pd.DataFrame
            Signature gene sets for each annotated cell type

        win_loglib : float
            Log-library size smoothed with neighboring spots
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
    
    def inference(self, x):
        # l is inferred from logrithmized x

        x_n = torch.log(1+x)
        hidden = self.l_enc(x_n)
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)
        ql = torch.clamp(ql, min = 0.01)

        # x is processed by dividing the inferred library
        x_n = torch.log(1+x)
        hidden = self.c_enc(x_n)

        qc_m =  self.c_enc_m(hidden)
        qc = Dirichlet(self.alpha * qc_m).rsample()[:,:,None]
        hidden = self.z_enc(x_n)
        qz_m_ct = self.z_enc_m(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_m_ct = (qc * qz_m_ct)
        
        qz_m = qz_m_ct.sum(axis=1)
        qz_logv_ct = self.z_enc_logv(hidden).reshape([x_n.shape[0],self.c_kn,self.c_bn])
        qz_logv_ct = (qc * qz_logv_ct)
        
        qz_logv = qz_logv_ct.sum(axis=1)
        qz = self.reparameterize(qz_m, qz_logv)

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
    
    def generative(
        self,
        inference_outputs,
        xs_k,
    ):
        
        qz = inference_outputs['qz']
        ql = inference_outputs['ql']

        hidden = self.px_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        px_rate = torch.exp(ql) * px_scale 
        
        xs_k = xs_k / torch.exp(ql) * torch.exp(ql.mean(axis=1,keepdims=True))
        pc_p = self.alpha * xs_k + 1e-5

        return dict(
                    px_rate=px_rate,
                    px_r=self.px_r,
                    pc_p=pc_p,
                    xs_k=xs_k,

                   )
    
    
    def get_loss(
        self,
        generative_outputs,
        inference_outputs,
        x,
        x_peri,
        library,
        device
    ):
    
        qc_m = inference_outputs["qc_m"]

        qz_m = inference_outputs["qz_m"]
        qz_logv = inference_outputs["qz_logv"]

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
        
        
        if (x_peri[:,0] == 1).sum() > 0:
            kl_divergence_c = kl(Dirichlet(qc_m[x_peri[:,0]==1]*self.alpha), Dirichlet(pc_p[x_peri[:,0]==1])).mean()
            if (x_peri[:,0] == 0).sum() > 0:
                # para-dependent
                if ((x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))).sum()>0:
                    kl_divergence_c = kl_divergence_c +  1e-1*kl(Dirichlet(qc_m[(x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))]*self.alpha), Dirichlet(pc_p[(x_peri[:,0]==0)&(library[:,0]<torch.quantile(self.win_loglib, 0.2))])).mean()
                if ((x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))).sum()>0:
                    kl_divergence_c = kl_divergence_c +  1e-2*kl(Dirichlet(qc_m[(x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))]*self.alpha), Dirichlet(pc_p[(x_peri[:,0]==0)&(library[:,0]>=torch.quantile(self.win_loglib, 0.2))])).mean()
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


class AVAE_poe(nn.Module):
    """ simple model design ***
    p(x|z)=f(z)
    p(z|x)~N(0,1)
    q(z|x)~g(x)
    """

    def __init__(
        self,
        adata,
        gene_sig,
        patch_r,
        win_loglib,
    ) -> None:
        """
        Auxiliary Variational AutoEncoder (AVAE) with Joint H&E inference
        - Core model for spatial deconvolution w/ H&E image integration

        Paramters
        ---------
        adata : sc.AnnData
            ST raw expression count (dim: [S, G])

        gene_sig : pd.DataFrame
            Signature gene sets for each annotated cell type

        patch_r : int
            Mini-patch size sampled around each spot from raw H&E image

        win_loglib : float
            Log-library size smoothed with neighboring spots
        """
        super(AVAE_poe, self).__init__()

        self.win_loglib = torch.Tensor(win_loglib)

        self.c_in = adata.shape[1]  # c_in : Num. input features (# input genes)
        self.c_bn = 10  # c_bn : latent number, numbers of bottle neck
        self.c_hidden = 256
        self.patch_r = patch_r
        self.c_kn = gene_sig.shape[1]

        self.alpha = torch.nn.Parameter(torch.rand(self.c_kn) * 1e3, requires_grad=True)

        self.c_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )

        self.c_enc_m = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_kn, bias=True),
            nn.BatchNorm1d(self.c_kn, momentum=0.01, eps=0.001),
            # nn.ReLU(),
            nn.Softmax()
            # nn.Softplus()
        )
        self.l_enc = nn.Sequential(
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
            # nn.Linear(self.c_hidden, 1, bias=True),
            # nn.ReLU(),
        )

        self.l_enc_m = nn.Linear(self.c_hidden, 1)
        self.l_enc_logv = nn.Linear(self.c_hidden, 1)

        # neural network f1 to get the z, p(z|x), f1(x,\phi_1)=[z_m,torch.exp(z_logv)]
        self.z_enc = nn.Sequential(
            # nn.Linear(self.c_in+self.c_kn, self.c_hidden, bias=True),
            nn.Linear(self.c_in, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )
        self.z_enc_m = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        self.z_enc_logv = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)

        # gene dispersion
        self.px_r = torch.nn.Parameter(torch.randn(self.c_in), requires_grad=True)
        # self.alpha= torch.nn.Parameter(torch.rand(1)*1e3,requires_grad=True)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.px_hidden_decoder = nn.Sequential(
            nn.Linear(self.c_bn, self.c_hidden, bias=True),
            nn.ReLU(),
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in),
            nn.Softplus(),
            # nn.Softmax(dim=-1),
            # nn.Softmax(),
        )

        self.px_r_poe = torch.nn.Parameter(torch.randn(self.c_in), requires_grad=True)

        self.img_l_enc = nn.Sequential(
            nn.Linear(self.patch_r * self.patch_r * 4 * 3, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )
        self.img_l_enc_m = nn.Linear(self.c_hidden, 1)
        self.img_l_enc_logv = nn.Linear(self.c_hidden, 1)

        self.img_c_enc = nn.Sequential(
            nn.Linear(self.patch_r * self.patch_r * 4 * 3, self.c_hidden, bias=True),  # flatten the images into 1D
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU()
        )
        self.img_c_enc_m = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_kn, bias=True),
            nn.BatchNorm1d(self.c_kn, momentum=0.01, eps=0.001),
            nn.Softmax()
        )

        self.imgVAE_z_enc = nn.Sequential(
            nn.Linear(self.patch_r * self.patch_r * 4 * 3, self.c_hidden, bias=True),
            nn.BatchNorm1d(self.c_hidden, momentum=0.01, eps=0.001),
            nn.ReLU(),
        )

        self.imgVAE_mu = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)
        self.imgVAE_logvar = nn.Linear(self.c_hidden, self.c_bn * self.c_kn)

        self.imgVAE_z_fc = nn.Linear(self.c_bn, self.c_hidden)

        self.imgVAE_dec = nn.Sequential(nn.Linear(self.c_hidden, self.patch_r * self.patch_r * 4 * 3, bias=True),
                                        nn.BatchNorm1d(self.patch_r * self.patch_r * 4 * 3, momentum=0.01, eps=0.001),
                                        )

        # PoE
        self.POE_z_fc = nn.Linear(self.c_bn, self.c_hidden)

        # neural network g to get the x_m and x_v, p(x|z), g(z,\phi_3)=[x_m,x_v]
        self.POE_z_fc = nn.Sequential(
            nn.Linear(self.c_bn, self.c_hidden, bias=True),
            nn.ReLU(),
        )

        self.POE_px_scale_decoder = nn.Sequential(
            nn.Linear(self.c_hidden, self.c_in),
            # nn.ReLU(),
            # nn.Softmax(),
            nn.Softplus()
        )

        self.POE_dec_img = nn.Sequential(
            nn.Linear(self.c_hidden, self.patch_r * self.patch_r * 4 * 3, bias=True),
            nn.BatchNorm1d(self.patch_r * self.patch_r * 4 * 3, momentum=0.01, eps=0.001),
            # nn.ReLU(),
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def inference(self,
                  x,
                  ):
        # library = torch.log(x.sum(1)).unsqueeze(1)
        # l is inferred from logrithmized x
        x1 = torch.log(1 + x)
        hidden = self.l_enc(x1)
        ql_m = self.l_enc_m(hidden)
        ql_logv = self.l_enc_logv(hidden)
        ql = self.reparameterize(ql_m, ql_logv)
        ql = torch.clamp(ql, min=0.1)

        x_n = torch.log(1 + x)
        hidden = self.c_enc(x_n)
        qc_m = self.c_enc_m(hidden)

        qc = Dirichlet(self.alpha * qc_m).rsample()[:, :, None]

        hidden = self.z_enc(x_n)
        qz_m_ct = self.z_enc_m(hidden).reshape([x1.shape[0], self.c_kn, self.c_bn])
        # qz_m_ct = (qc * qz_m_ct)

        qz_m = (qc * qz_m_ct).sum(axis=1)
        qz_logv_ct = self.z_enc_logv(hidden).reshape([x1.shape[0], self.c_kn, self.c_bn])
        # qz_logv_ct = (qc * qz_logv_ct)
        qz_logv = (qc * qz_logv_ct).sum(axis=1)

        qz = self.reparameterize(qz_m, qz_logv)
        return dict(
            qc_m=qc_m,
            qc=qc,
            qz_m=qz_m,
            qz_m_ct=qz_m_ct,
            qz_logv=qz_logv,
            qz_logv_ct=qz_logv_ct,
            qz=qz,
            ql_m=ql_m,
            ql_logv=ql_logv,
            ql=ql
        )

    def predict_imgVAE(self, x):

        x = x * 255
        hidden = self.img_l_enc(x)
        img_ql_m = self.img_l_enc_m(hidden)
        img_ql_logv = self.img_l_enc_logv(hidden)
        img_ql = self.reparameterize(img_ql_m, img_ql_logv)
        img_ql = torch.clamp(img_ql, min=0.1)

        hidden = self.img_c_enc(x)
        img_qc_m = self.img_c_enc_m(hidden)
        img_qc = Dirichlet(self.alpha * img_qc_m).rsample()[:, :, None]

        hidden = self.imgVAE_z_enc(x)
        img_qz_m_ct = self.imgVAE_mu(hidden).reshape([x.shape[0], self.c_kn, self.c_bn])
        # img_qz_m_ct = (img_qc * img_qz_m_ct)
        img_qz_m = (img_qc * img_qz_m_ct).sum(axis=1)

        img_qz_logv_ct = self.imgVAE_logvar(hidden).reshape([x.shape[0], self.c_kn, self.c_bn])
        # img_qz_logv_ct = (img_qc * img_qz_logv_ct)
        img_qz_logv = (img_qc * img_qz_logv_ct).sum(axis=1)

        img_qz = self.reparameterize(img_qz_m, img_qz_logv)

        hidden = self.imgVAE_z_fc(img_qz)
        reconstruction = self.imgVAE_dec(hidden)

        return dict(reconstruction=reconstruction,
                    img_z_mu=img_qz_m,
                    img_z_logv=img_qz_logv,
                    img_qz_m_ct=img_qz_m_ct,
                    img_qz_logv_ct=img_qz_logv_ct,
                    img_qc=img_qc,
                    img_qc_m=img_qc_m,
                    img_ql_m=img_ql_m,
                    img_ql_logv=img_ql_logv,
                    img_ql=img_ql
                    )

    def generative(
        self,
        inference_outputs,
        xs_k,
        img_path_outputs
    ):

        qz = inference_outputs['qz']
        ql = inference_outputs['ql']
        ql_m = inference_outputs['ql_m']
        img_ql = img_path_outputs['img_ql']

        hidden = self.px_hidden_decoder(qz)
        px_scale = self.px_scale_decoder(hidden)
        px_rate = torch.exp((ql + img_ql) / 2) * px_scale

        xs_k = xs_k / torch.exp(ql) * torch.exp(ql.mean(axis=1, keepdims=True))

        pc_p = self.alpha * xs_k + 1e-5

        return dict(
            px_rate=px_rate,
            px_r=self.px_r,
            pc_p=pc_p,
            xs_k=xs_k,
        )

    def predictor_POE(
        self,
        inference_outputs,
        exp_path_outputs,
        img_path_outputs,
    ):

        mu_img = img_path_outputs['img_z_mu']
        logvar_img = img_path_outputs['img_z_logv']

        mu_exp = inference_outputs['qz_m']
        logvar_exp = inference_outputs['qz_logv']
        ql = inference_outputs['ql']

        batch, _ = mu_exp.shape

        var_poe = torch.div(1.,
                            1 +
                            torch.div(1., torch.exp(logvar_exp)) +
                            torch.div(1., torch.exp(logvar_img))
                            )

        mu_poe = var_poe * (0 +
                            mu_exp * torch.div(1., torch.exp(logvar_exp) + 1e-5) +
                            mu_img * torch.div(1., torch.exp(logvar_img) + 1e-5)
                            )

        z = self.reparameterize(mu_poe, torch.log(var_poe + 0.001))

        hidden = self.POE_z_fc(z)

        px_scale_rna = self.POE_px_scale_decoder(hidden)

        px_rate_poe = torch.exp(ql) * px_scale_rna
        px_r_poe = self.px_r_poe
        # reconstruction_rna_peri =  self.POE_dec_rna(torch.cat((x_peri,x_peri_sig),1))

        reconstruction_img = self.POE_dec_img(hidden)

        return dict(px_rate_poe=px_rate_poe,
                    px_r_poe=px_r_poe,
                    reconstruction_img=reconstruction_img,
                    mu_poe=mu_poe,
                    var_poe=torch.log(var_poe + 0.001)
                    )

    def get_loss(
        self,
        generative_outputs,
        inference_outputs,
        img_path_outputs,
        poe_path_outputs,
        x,
        x_peri,
        library,
        adata_img
    ):

        beta = 0.001
        alpha_c = 5

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

        px_rate_poe = poe_path_outputs['px_rate_poe']
        px_r_poe = poe_path_outputs['px_r_poe']
        recon_poe_img = poe_path_outputs['reconstruction_img']
        logvar_poe = poe_path_outputs['var_poe']
        mu_poe = poe_path_outputs['mu_poe']

        criterion = nn.MSELoss(reduction='sum')
        reconst_loss_poe_rna = -NegBinom(px_rate_poe, torch.exp(px_r_poe)).log_prob(x).sum(-1).mean()

        bce_loss_poe_img = criterion(recon_poe_img, adata_img)

        Loss_IBJ = (torch.sum(reconst_loss_poe_rna) + bce_loss_poe_img) + \
                   beta * (-0.5 * torch.sum(1 + logvar_poe - mu_poe.pow(2) - logvar_poe.exp()))

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_logv)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(torch.exp(qz_logv))), Normal(mean, scale)).sum(
            dim=1
        ).mean()

        kl_divergence_n = kl(Normal(ql_m, torch.sqrt(torch.exp(ql_logv))), Normal(library, torch.ones_like(ql))).sum(
            dim=1
        ).mean()

        if (x_peri[:, 0] == 1).sum() > 0:

            kl_divergence_c = kl(Dirichlet(qc_m[x_peri[:, 0] == 1] * self.alpha),
                                 Dirichlet(pc_p[x_peri[:, 0] == 1])).mean()
            if (x_peri[:, 0] == 0).sum() > 0:
                if ((x_peri[:, 0] == 0) & (library[:, 0] < torch.quantile(self.win_loglib, 0.2))).sum() > 0:
                    kl_divergence_c = kl_divergence_c + 1e-1 * kl(Dirichlet(qc_m[(x_peri[:, 0] == 0) & (
                                library[:, 0] < torch.quantile(self.win_loglib, 0.2))] * self.alpha), Dirichlet(
                        pc_p[(x_peri[:, 0] == 0) & (library[:, 0] < torch.quantile(self.win_loglib, 0.2))])).mean()
                if ((x_peri[:, 0] == 0) & (library[:, 0] >= torch.quantile(self.win_loglib, 0.2))).sum() > 0:
                    kl_divergence_c = kl_divergence_c + 1e-2 * kl(Dirichlet(qc_m[(x_peri[:, 0] == 0) & (
                                library[:, 0] >= torch.quantile(self.win_loglib, 0.2))] * self.alpha), Dirichlet(
                        pc_p[(x_peri[:, 0] == 0) & (library[:, 0] >= torch.quantile(self.win_loglib, 0.2))])).mean()
        else:
            kl_divergence_c = torch.Tensor([0.0])

        reconst_loss = -NegBinom(px_rate, torch.exp(px_r)).log_prob(x).sum(-1).mean()

        loss_exp = reconst_loss + kl_divergence_z + kl_divergence_c + kl_divergence_n

        mu_img = img_path_outputs['img_z_mu']
        logvar_img = img_path_outputs['img_z_logv']
        recon_img = img_path_outputs['reconstruction']
        img_qc = img_path_outputs['img_qc']
        img_ql = img_path_outputs['img_ql']
        img_ql_m = img_path_outputs['img_ql_m']
        img_ql_logv = img_path_outputs['img_ql_logv']
        img_qc_m = img_path_outputs['img_qc_m']

        kl_divergence_n_img = kl(Normal(img_ql_m, torch.sqrt(torch.exp(img_ql_logv))),
                                 Normal(library, torch.ones_like(ql))).sum(
            dim=1
        ).mean()

        kl_divergence_z_img = kl(Normal(mu_img, torch.sqrt(torch.exp(logvar_img))), Normal(mean, scale)).sum(
            dim=1
        )

        if (x_peri[:, 0] == 1).sum() > 0:
            kl_divergence_c_img = kl(Dirichlet(img_qc_m[x_peri[:, 0] == 1] * self.alpha),
                                     Dirichlet(pc_p[x_peri[:, 0] == 1])).mean()
        else:
            kl_divergence_c_img = torch.Tensor([0.0])

        bce_loss_img = criterion(recon_img, adata_img)
        loss_img = torch.sum(bce_loss_img + kl_divergence_z_img + kl_divergence_n_img + kl_divergence_c_img)

        Loss_IBM = (loss_exp + loss_img)
        loss = Loss_IBJ + alpha_c * Loss_IBM

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
        inference_outputs = model.inference(x)

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


def model_ct_exp(
    model,
    adata_sample,
    sig_mean,
    device,
    library_i,
    lib_low,
    ct_idx
):
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
