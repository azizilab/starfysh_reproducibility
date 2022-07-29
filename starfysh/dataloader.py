import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset


class VisiumDataset(Dataset):
    """
    Loading preprocessed Visium AnnData, gene signature & Anchor spots for Starfysh training
    """

    def __init__(self, adata, gene_sig_exp_m, adata_pure, library_n):
        spots = adata.obs_names
        genes = adata.var_names
        
        x = adata.X if isinstance(adata.X, np.ndarray) else adata.X.A
        #self.ci = adata.obs['c_i']
        self.expr_mat = pd.DataFrame(x, index=spots, columns=genes)
        #self.gsva = gsva_score
        self.gene_sig_exp_m = gene_sig_exp_m
        self.adata_pure = adata_pure
        self.library_n = library_n

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )

        return (sample, 
                #torch.Tensor(self.ci[idx]), 
                #torch.Tensor(self.gsva.iloc[idx,:]),
                torch.Tensor(self.gene_sig_exp_m.iloc[idx,:]),
                torch.Tensor(self.adata_pure[idx,:]),
                torch.Tensor(self.library_n[idx,None]),
               )
    