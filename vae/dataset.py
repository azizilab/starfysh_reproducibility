import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VisiumDataset(Dataset):

    def __init__(self, adata):
        spots = adata.obs_names
        genes = adata.var_names
        x = adata.X.A

        self.expr_mat = pd.DataFrame(x, index=spots, columns=genes)

    def __len__(self):
        return len(self.expr_mat)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.Tensor(
            np.array(self.expr_mat.iloc[idx, :], dtype='float')
        )

        return sample

