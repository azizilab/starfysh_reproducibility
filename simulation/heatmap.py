import pandas as pd
import numpy as np

import os
import argparse as arp

import matplotlib.font_manager
from matplotlib import rcParams

font_list = []
fpaths = matplotlib.font_manager.findSystemFonts()
for i in fpaths:
    try:
        f = matplotlib.font_manager.get_font(i)
        font_list.append(f.family_name)
    except RuntimeError:
        pass

font_list = set(font_list)
plot_font = 'Liberation Sans'

rcParams['font.family'] = plot_font
rcParams.update({'font.size': 10})
rcParams.update({'figure.dpi': 300})
rcParams.update({'figure.figsize': (3,3)})
rcParams.update({'savefig.dpi': 500})

import warnings
warnings.filterwarnings('ignore')

import utils

starfysh_wu = {
    "Activated CD8": ["T_cells_c7_CD8+_IFNG"],
    "B cells memory": ["B cells Memory"],
    "Basal": ["Cancer Basal SC"],
    "CAFs MSC iCAF-like": ["CAFs MSC iCAF-like s1", "CAFs MSC iCAF-like s2"],
    "CAFs myCAF-like": ["CAFs myCAF like s4", "CAFs myCAF like s5"],
    "Endothelial": ["Endothelial ACKR1", "Endothelial CXCL12", "Endothelial Lymphatic LYVE1", "Endothelial RGS5"],
    "Macrophage M2": ["Myeloid_c9_Macrophage_2_CXCL10"],
    "PVL immature": ["PVL Immature s1", "PVL_Immature s2"],
    "Tcm": ["T_cells_c0_CD4+_CCR7"],
    "pDC": ["Myeloid_c4_DCs_pDC_IRF7"]
}

def plot_heatmap(y_true, y_pred, out_dir, name):

    y_pred_final = pd.DataFrame(index=y_pred.index)
    for key in starfysh_wu:
        y_pred_final[key] = y_pred[starfysh_wu[key]].sum(axis=1)
    y_pred_final

    utils.disp_corr(y_true, y_pred_final, outdir=out_dir, filename="corr_matrix", savefig=True, format="svg", title=name, fontsize=5)

def main():
    prs = arp.ArgumentParser()
    prs.add_argument(
        '-n', '--name',
        type=str, required=True,
        help="name of model"
    )
    prs.add_argument(
        '-t', '--y_true', 
        type=str, required=True,
        help="path to ground-truth proportions csv file"
    )
    prs.add_argument(
        '-p', '--y_pred',
        type=str, required=True,
        help="path to predicted proportions csv file"
    )
    prs.add_argument(
        '-o', '--out_dir',
        type=str, default="./",
        help="path to output directory"
    )

    args = prs.parse_args()

    name = args.name
    y_true = pd.read_csv(args.y_true, index_col=0)
    y_pred = pd.read_csv(args.y_pred, index_col=0)
    out_dir = args.out_dir

    plot_heatmap(y_true, y_pred, out_dir, name)

if __name__ == '__main__':
    main()
