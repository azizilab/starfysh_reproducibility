# -*- coding: utf-8 -*-
"""kaylee_spatial_sim.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TyT0txSz1wC5HBjOnPVkJ9ruiXfsGRKC
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install pymc3
# !pip install scanpy
# !pip install gpytorch

import os
import os.path as osp
import argparse as arp
import sys
import math
import json

import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm

import scanpy as sc
import torch
import gpytorch
import matplotlib.pyplot as plt

from scipy.io import mmread

from skimage import io
import json

from scipy.stats import lognorm, poisson, gamma


def sample(meta_df, cell_type, k_cells=9, specs=None):
  """
  meta_df: metadata df with cell types
  cell_type: e.g. minor or major
  k_cells: int k cells to be sampled
  specs: specific cell names to be sampled
  """

  sample = []

  # filtering and choosing cell types to simulate
  if specs != None:
    assert (isinstance(specs, list) or isinstance(specs, np.ndarray)), "Please ensure that specs is iterable"
    assert (set(np.intersect1d(meta_df[cell_type], specs)) == set(specs)), "Please ensure input cell types (specs) are valid names"
    sample = specs

  else:
    try:
      sample = meta_df[cell_type].unique()
    except KeyError:
      print("invalid cell type")  # or throw error? this part doesn't work, fix later
      return -1
    sample = np.random.choice(sample, k_cells, replace=False)

  return list(sample)

# Code from cell2location simulations
# Reference: https://github.com/vitkl/cell2location_paper

def kernel(X1, X2, l=1.0, eta=1.0):
  """
  Isotropic squared exponential kernel. Computes 
  a covariance matrix from points in X1 and X2.
  
  Args:
      X1: Array of m points (m x d).
      X2: Array of n points (n x d).

  Returns:
      Covariance matrix (m x n).
  """
  sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
  return eta**2 * np.exp(-0.5 / l**2 * sqdist)
  
def generate_grid(n=[50, 50]):
  n1, n2 = n 
  x1 = np.linspace(0, 100, n1)[:,None] #saptial dimensions 
  x2 = np.linspace(0, 100, n2)[:,None] #saptial dimensions 

  # make cartesian grid out of each dimension x1 and x2
  return pm.math.cartesian(x1[:,None], x2[:,None]), x1, x2

def random_GP(X, 
            x1=1, x2=1, #coordinates
            n_variables = 5, # zones
            eta_true = 5, #variance, defines overlapping
            l1_true=[8, 10, 15], #bw parameter
            l2_true=[8, 10, 15]
            ):

  #cov1, cov2 = kernel(x1, x1, l=l1_true), kernel(x2, x2, l=l2_true)
  K = [np.kron(kernel(x1, x1, l=l1_true[i], eta=eta_true), 
                kernel(x2, x2, l=l2_true[i], eta=eta_true)) 
        for i in range(n_variables)]
  
  #samples from GP
  gaus_true = np.stack([
      np.random.multivariate_normal(np.zeros(X.shape[0]), 2*K[i]) 
      for i in range(n_variables)
  ]).T 
  
  N_true = (np.exp(gaus_true).T / np.exp(gaus_true).sum(axis=1)).T #softmax transform 
  return N_true

def sample_GP(locations, 
            ct_labels,
            n_cells,
            bw,
            x1, 
            x2,
            eta=2
            ):
  """
  Sample abundances with GP
  
      Higher `eta` values represent higher transition rates
      from high-density <--> low -density region
      
  """
  abundances = random_GP(
      X=locations, 
      x1=x1, 
      x2=x2, 
      n_variables=n_cells, 
      eta_true=eta,
      l1_true=bw,
      l2_true=bw
  )
  
  abundances = abundances / abundances.max(0)
  abundances[abundances < 0.1] = 0
  
  return pd.DataFrame(
      abundances, 
      index=[f'location_{i}' for i in range(abundances.shape[0])],
      columns=ct_labels
  )


def plot_spatial(values, n=[50,50], nrows=5, names=['cell type'],
                vmin=0, vmax=1):
  
  n_cell_types = values.shape[1]
  n1, n2 = n 
  ncols = np.ceil((n_cell_types+1) / nrows).astype(np.uint16)
  for ct in range(n_cell_types):
      try:
          plt.subplot(nrows, ncols, ct+1)
          plt.imshow(values[:,ct].reshape(n1,n2).T, 
                      cmap=plt.cm.get_cmap('magma'),
                      vmin=vmin, vmax=vmax
                    )
          plt.colorbar()
          if len(names) > 1:
              plt.title(names[ct])
          else:
              plt.title(f'{names[0]} {ct+1}')
              
      except ValueError:
          continue

  try:
      plt.subplot(nrows, ncols, n_cell_types+1) 
      plt.imshow(values.sum(axis=1).reshape(n1,n2).T, 
                  cmap=plt.cm.get_cmap('Greys'))
      plt.colorbar()
      plt.title('total')
  
  except ValueError as e:
      print(e)

def get_ct_name(ct_map, idxs):
  try:
      return [ct_map[i] for i in idxs]
  except KeyError(e):
      print('Index {} not included in cell type map'.format(i))
      return -1

def _random_assign(n):
  """Random one-to-one assignemnt by permuting an NxN identity matrix"""
  mat = np.random.permutation(np.diag(np.ones(n)))
  return mat

def assign_ct_pattern(cell_types, n_patterns):
  """
  Assign cell-type categories to spatial pattern categories
  """
  mat = _random_assign(n_patterns)
  
  patterns_df = pd.DataFrame(mat, index=cell_types) 
  patterns_df.index.name='cell_types'
  
  return patterns_df

def pattern_spot(cell_types, ct_names, n_locations=[50,50], n_experiments=1, mean=12):
  ''' Generate per-spot spatial pattern '''

  # Generate matrix of which cell types are in which zones
  # Specify number of distinct spatial patterns under each category

  np.random.seed(1001)

  ct_patterns_df = assign_ct_pattern(cell_types, len(cell_types))
  mean_var_ratio = 1.5
  mean = 12   # assuming specific cell type
  bandwidth = np.random.gamma(
      mean * mean_var_ratio, 1 / mean_var_ratio,
      size= len(cell_types)
  )

  locations_1, x1, x2 = generate_grid(n=n_locations)
  locations = np.concatenate([locations_1 for _ in range(n_experiments)], axis=0)

  # Sample per-cell-type, per-spot abundance
  abundances_df = pd.DataFrame()
  for e in range(n_experiments):
      
      abundances_df_1 = sample_GP(
          locations=locations_1, 
          ct_labels=cell_types,
          n_cells = len(cell_types),
          bw = bandwidth,
          x1=x1, x2=x2
      )

      abundances_df_1.index = [f'exper{e}_{l}' for l in abundances_df_1.index]
      abundances_df = pd.concat((abundances_df, abundances_df_1), axis=0)

  # generate per-spot spatial pattern
  cell_abundances = np.dot(abundances_df, ct_patterns_df.T)
  q_sf = np.tile(
      np.random.lognormal(0.5, 0.5, size=len(cell_types)),
      reps=(n_locations[0] * n_locations[1], 1),
  )

  cell_abundances = cell_abundances * q_sf
  cell_abundances_df = pd.DataFrame(
      cell_abundances, 
      index=abundances_df.index,
      columns=ct_patterns_df.index
  )

  cell_abundances_df.columns = ct_names

  cell_count_df = np.ceil(cell_abundances_df).astype(np.uint16)
  proportion_df = cell_count_df / np.expand_dims(cell_count_df.sum(1), 1)

  return cell_abundances_df, cell_count_df, proportion_df

def find_nonzero_cts(df):
  """
  Find expressed (non-zero) cell types per spot
  """
  cols = df.columns
  return (df > 0).apply(
      lambda x: list(cols[x.values]), axis=1
  )


def find_sc_index(df, cell_type, libsize_raw, l, n_nbr=3):
  """
  Find the scRNA-seq cell with close library size to given expected library size of 
  the synthetic spot (s). 
  
  Either randomly sample the index out of n closest libsize or 
  deterministically choose the cell index with the closest libsize
  
  Returns
  -------
  expr : list
      Gene expression of the selected scRNA-seq cell
  """
  
  # Bug: the index in subsetted libsize different from the original count index!!!
  cell_types = np.unique(df.index.get_level_values(1))
  assert cell_type in cell_types, 'Unexpected cell type {}'.format(cell_type)
  
  # 'Mask out' indices other than the given cell type
  mask = df.index.get_level_values(1) == cell_type
  libsize = libsize_raw.copy()
  libsize[~mask] = -np.inf   
  
  n = np.random.randint(n_nbr) if n_nbr > 0 else 0
  idx = np.abs(libsize - l).argsort()[n]
  expr = df.iloc[idx].to_numpy()
  
  return expr


def fit_st_libsize(libsize, idxs):
  """
  Fit library size from scRNA-seq data & sample expected library size 
  to each synthetic Spatial Transcriptomics spot
  """
  
  s, loc, scale = lognorm.fit(libsize)
  l = pd.Series(
      lognorm.rvs(s, loc=loc, scale=scale, size=len(idxs)),
      index=idxs
  )
  
  return np.round(l).astype(np.uint32)
  

def simulate_spots(df, count_df, expr_factors, l, ratio=0.16, verbose=False):
  """
  Simulate spot x gene synthetic matrix given 
  (1). per-cell density; (2). library size for each spot
  """
  n_spot = count_df.shape[0]
  n_genes = df.shape[1]
  libsize = df.sum(1).to_numpy().astype(np.float64)
  
  sims = np.zeros((n_spot, n_genes))
  
  for i, cell_types in enumerate(expr_factors):
      if verbose and i % 100 == 0:
          print('Simulating spot {}...'.format(i))
          
      l_s = l[i] # expected library size for given spot
      spot_expr = np.zeros(n_genes)
      
      cell_count = count_df[cell_types].iloc[i]
      n_cells = cell_count.sum().astype(np.uint32)
      
      # Sample expected library size to each single-cell index 
      # captured to current spot
      # for now fix the same expected library size for all scRNA-seq to the given spot
      """
      expected_libsize = poisson.rvs(
          l[i], 
          size=n_cells
      )
      """
      
      cell_type_freqs = np.hstack([
          np.repeat(cell_type, count)
          for (cell_type, count) in zip(cell_types, cell_count)
      ])
      
      # for count, cell_type in zip(expected_libsize, cell_type_freqs):
      for cell_type in cell_type_freqs:
          ct_expr = find_sc_index(df, cell_type, libsize, l_s, n_nbr=0)
          spot_expr += ct_expr
          
      spot_expr = np.round(spot_expr * ratio).astype(np.uint64)
      sims[i] = spot_expr
      
  sims_df = pd.DataFrame(
      sims,
      index=count_df.index,
      columns=df.columns
  )
  
  return sims_df

def sim_driver(df_sc, adata, cell_count_df, cell_abundances_df):
  libsize = df_sc.values.sum(1)
  libsize_st = adata.X.A.sum(1)

  cap_ratio = np.median(np.log1p(libsize_st)) / np.median(np.log1p(libsize))
  cap_ratio_per_spot = cap_ratio / cell_count_df.sum(1).mean()
  cap_ratio_per_spot

  # Get list of expressed (non-zero) cell type for each spot
  expressed_factors = find_nonzero_cts(cell_count_df)

  # Fit library size to each synthetic spot
  l_s = fit_st_libsize(
      libsize=df_sc.values.sum(1), 
      idxs=cell_abundances_df.index
  )

  synth_df = simulate_spots(
      df_sc,
      cell_count_df,
      expressed_factors,
      l_s,
      ratio=cap_ratio_per_spot,
      verbose=True
  )

  return synth_df

def main():
  prs = arp.ArgumentParser()
  prs.add_argument('-r','--sc_path',
                   type = str,
                   required = True,
                   help = "path to scrna data directory")
  prs.add_argument('-s', '--spatial_path',
                   type = str,
                   required = True,
                   help = "path to spatial data directory")
  prs.add_argument('-i', '--sample_id',
                   type = str,
                   required = True,
                   help = "sample id")
  prs.add_argument('-nc', '--n_celltypes',
                   type = int, 
                   default = 9,
                   help = "number of cell types to sample (n >= 9)")
  prs.add_argument('-ct', '--cell_type',
                   type = str,
                   default = 'minor', 
                   help = "cell type (major, minor, subset) to sample")
  prs.add_argument('-o', '--out_dir', 
                   required = True,
                   help = "output directory")
  
  args = prs.parse_args()

  out_dir = args.out_dir
  if not osp.exists(out_dir):
        os.mkdir(out_dir)
  
  sc_path = args.sc_path
  spatial_path = args.spatial_path
  sample_id = args.sample_id
  cell_type = 'celltype_' + args.cell_type

  meta_df = pd.read_csv(os.path.join(sc_path, 'metadata.csv'), index_col=[0])
  barcode_df = pd.read_csv(
      os.path.join(sc_path, 'count_matrix_barcodes.tsv'), 
      delimiter='\t',
      header=None # this file doesn't contain the true "header" (column info), the first row is a TRUE barcode
      )
  gene_df = pd.read_csv(
      os.path.join(sc_path, 'count_matrix_genes.tsv'),
      delimiter='\t',
      header=None # this file doesn't contain the true "header" (column info), the first row is a TRUE gene
      )
  cnt = mmread(
      os.path.join(sc_path, 'count_matrix_sparse.mtx')
      ).toarray()
  barcodes = pd.DataFrame(barcode_df[0].values, index=barcode_df[0].values, columns=['Barcode'])
  genes = pd.DataFrame(gene_df[0].values, index=gene_df[0].values, columns=['features'])
  adata_sc = sc.AnnData(
      X=cnt.T,
      obs=barcodes,
      var=genes
      )
  
  df_sc = adata_sc.to_df()

  annots = meta_df[cell_type]
  df_sc.index.name = 'Barcode'
  df_sc['cell_type'] = annots.to_list()
  df_sc.set_index('cell_type', inplace=True, append=True)

  adata = sc.read_h5ad(
      os.path.join(spatial_path, sample_id+'.h5ad')
  )

  adata.var_names_make_unique()

  ct_names = sample(meta_df, cell_type)
  cts = [i for (i, ct) in enumerate(ct_names)]
  cell_abundances_df, cell_count_df, proportion_df = pattern_spot(cts, ct_names)

  synth_df = sim_driver(df_sc, adata, cell_count_df, cell_abundances_df)

  synth_df.to_csv(
      os.path.join(out_dir, 'counts.st_synth.csv'),
      index=True
  )
  cell_count_df.to_csv(
      os.path.join(out_dir, 'members.st_synth.csv'),
      index=True
  )
  proportion_df.to_csv(
      os.path.join(out_dir, 'proportions.st_synth.csv'),
      index=True
  )

if __name__ == '__main__':
  main()
