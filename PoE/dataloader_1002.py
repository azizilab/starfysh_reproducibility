# datasets
from torch.utils.data import DataLoader
import torchvision 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np

class shenet_traindatastack(torch.utils.data.Dataset):
    """
    return the data stack with expression and image 
    """

    def __init__(self, train_data,map_info,patch_r):

        super(shenet_traindatastack, self).__init__()
        self.genexp = train_data[0]
        self.genexp_sig = train_data[1]
        self.image = train_data[2]
        
        

        self.map_info =  map_info
        self.map_xlen = map_info.loc[:,'array_col'].max()-map_info.loc[:,'array_col'].min()+1
        self.map_ylen = map_info.loc[:,'array_row'].max()-map_info.loc[:,'array_row'].min()+1
        self.spot_img_stack = []
        for i in range(len(self.genexp)):
            img_xmin = int(self.map_info.iloc[i]['imagecol'])-patch_r
            img_xmax = int(self.map_info.iloc[i]['imagecol'])+patch_r
            img_ymin = int(self.map_info.iloc[i]['imagerow'])-patch_r
            img_ymax = int(self.map_info.iloc[i]['imagerow'])+patch_r
            self.spot_img_stack.append(self.image[img_ymin:img_ymax,img_xmin:img_xmax])

    def __getitem__(self, idx):
        return np.array(self.genexp.iloc[idx]),np.array(self.genexp_sig.iloc[idx]),self.spot_img_stack[idx],self.map_info.index[idx]

    def __len__(self):
        return len(self.genexp)