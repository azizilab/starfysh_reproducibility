#!/usr/bin/env python
# Simulate synthetic Spatial Transcriptomics data from scRNA-seq dataset

import os
import numpy as np
import pandas as pd
import argparse

from scipy import io as scipy_io
from argparse import RawTextHelpFormatter
from progress.bar import ChargingBar


def load_sc(rc_file,
            bc_file=None,
            gene_file=None,
            header=None,
            index_col=None,
            sep=',',
            is_sparse=False,
            transpose=False
            ):
    """
    Load scRNA-seq dataset

    Parameters
    ----------
    rc_file : str
        Path to raw count matrix

    bc_file : str
        Path to barcode names (if data in sparse format)

    gene_file : str
        Path to gene names (if data in sparse format)\

    transpose : bool
        Whether raw counts is [Gene x Cell] and requires transpose
    """
    assert os.path.exists(rc_file) and os.path.isfile(rc_file), \
        "Raw count file doesn't exist"

    if is_sparse:
        assert isinstance(bc_file, str) and isinstance(gene_file, str), \
            "Sparse raw count requires corresponding metadata for barcode & gene names"
        assert os.path.exists(bc_file) and os.path.isfile(bc_file) and \
               os.path.exists(gene_file) and os.path.isfile(gene_file), \
            "scRNA-seq metadata files don't exist"

    if bc_file[-3:] == 'tsv' and gene_file[-3:] == 'tsv':
        sep = '\t'

    if is_sparse:
        df_bc = pd.read_csv(bc_file, sep=sep, header=header)
        df_genes = pd.read_csv(gene_file, sep=sep, header=header)
        rc = scipy_io.mmread(rc_file)
        rc = rc.A.T if transpose else rc.A
        df_sc = pd.DataFrame(rc, index=df_bc.iloc[:, 0], columns=df_genes.iloc[:, 0])

    else:
        df_sc = pd.read_csv(rc_file, sep=sep, index_col=index_col, header=header)

    return df_sc


def load_meta(meta_file,
              col_name='celltype_major',
              index_col=0):
    """Load scRNA-seq metadata w/ cell type anotations for simulation"""
    assert os.path.exists(meta_file) and os.path.isfile(meta_file), \
        "Metadata file doesn't exist"

    df_meta = pd.read_csv(meta_file, index_col=index_col)
    annots = df_meta[col_name]
    return annots


def gen_synth_st(df_sc,
                 annots,
                 n_spots,
                 n_capture=(5, 20),
                 dir_prior=(0.25, 0.5),
                 ):
    """
    Generate synthetic ST data from scRNA-seq data & cell-type annotations

    Parameters
    ----------
    df_sc : pd.DataFrame
        Input scRNA-seq raw count matrix (dim: [Cell, Gene])

    annots : array-like
        Cell-type annotations of scRNA-seq data

    n_spots : int
        Number of synthetic ST spots

    n_capture : tuple
        Range of possible # cell captured by each spot

    dir_prior : tuple(float, float)
        Randomness prior for Dirchlet parameter to control cell-type mixture sparsity

    Returns
    -------
    df_st : pd.DataFrame
        Synthetic Spatial Transcriptomics matrix (dim: [Spot, Gene])

    df_mix : pd.DataFrame
        Ground-truth cell-type count of each synthetic spot
    """
    cmin, cmax = n_capture
    assert 0 < cmin < cmax, "Invalid # cell capture range for each pesudospot"

    sigs = np.unique(annots)
    sig_dict = {sig: i for i, sig in enumerate(sigs)}

    st = np.zeros((n_spots, df_sc.shape[1]))
    decomp = np.zeros((n_spots, len(sigs)))
    st_barcodes = ['S' + str(i) for i in range(n_spots)]

    bar = ChargingBar('Simulating:', max=n_spots, suffix='%(percent)d%%')
    print('Start synthesizing pseudo-spots...')

    for i in range(n_spots):
        # Select `n_cell` samples with Dir(a) priors to mixture rate
        bar.next()
        n_cell = np.random.randint(cmin, cmax + 1)
        n_cps_float = np.random.dirichlet(np.ones(len(sigs)) * np.random.uniform(dir_prior[0], dir_prior[1])) * n_cell
        n_cps = n_cps_float.round().astype(np.int32)
        if n_cps.sum() == 0:
            n_cps = np.ones(n_cell)

        idxs = []
        for n, sig in zip(n_cps, sigs):
            rp = True if (annots == sig).sum() <= n else False
            idxs.extend(np.random.choice(annots[annots == sig].index, n, replace=rp))

        # Update synth. spot counts
        cap_rate = np.random.uniform(0.2, 1)
        st[i] = np.round(cap_rate * df_sc.loc[idxs].sum(0)).astype(np.int32)

        # Update ground-truth cell-type decomposition
        ct = annots.loc[idxs].value_counts().to_dict()
        decomp[i, list(map(sig_dict.get, ct.keys()))] = list(ct.values())

    bar.finish()
    print('Finished ST simulation\n')

    df_st = pd.DataFrame(st, index=st_barcodes, columns=df_sc.columns)
    df_mix = pd.DataFrame(decomp, index=st_barcodes, columns=sigs)

    return df_st, df_mix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic ST simulation options",
                                     formatter_class=RawTextHelpFormatter)
    # Input files
    parser.add_argument('--rc', dest='rc_file', type=str, action='store',
                        help='scRNA-seq raw counts')
    parser.add_argument('--barcode', dest='bc_file', type=str, default=None, action='store',
                        help='scRNA-seq barcode names')
    parser.add_argument('--gene', dest='gene_file', type=str, default=None, action='store',
                        help='scRNA-seq gene names')
    parser.add_argument('--meta', dest='meta_file', type=str, action='store',
                        help='scRNA-seq cell-type metadata')

    # Output files
    parser.add_argument('-e', '--expr', dest='expr_file', type=str, default=None, action='store',
                        help='Output ST expr file')
    parser.add_argument('-m', '--mix', dest='mix_file', type=str, default=None, action='store',
                        help='Output ST mixture file')

    # Simulation specs
    parser.add_argument('--n-spots', dest='n_spots', type=int, default=2000, action='store',
                        help='Num. synthetic ST spots')
    parser.add_argument('--n-cells', dest='n_cells', nargs='+', type=int, default=[5, 20], action='store',
                        help='Range of num. cells captured in each ST spot')
    parser.add_argument('--dir-prior', dest='dir_prior', nargs='+', type=float, default=[0.25, 0.5], action='store',
                        help='Range of cell-type proportion priors for simulation')

    parser.add_argument('--is-csv', dest='is_csv', action='store_true',
                        help='Whether input scRNA-seq data is sparse format (if is-csv - NO)')
    parser.add_argument('--transpose', dest='transpose', action='store_true',
                        help='Whether transpose input count matrix')
    parser.set_defaults(feature=True)

    args = parser.parse_args()
    assert len(args.n_cells) == 2, "Invalid length of cell capture range"
    assert len(args.dir_prior) == 2 and \
           0 <= args.dir_prior[0] < args.dir_prior[1] <= 1, \
        "Invalid cell-type porportion priors range"
    is_sparse = not args.is_csv

    # Load input files
    print('Loading input scRNA-seq data...')
    df_sc = load_sc(rc_file=args.rc_file,
                    bc_file=args.bc_file,
                    gene_file=args.gene_file,
                    is_sparse=is_sparse,
                    transpose=args.transpose
                    )
    annots = load_meta(args.meta_file)

    # Simulation
    df_st, df_mix = gen_synth_st(df_sc=df_sc,
                                 annots=annots,
                                 n_spots=args.n_spots,
                                 n_capture=tuple(args.n_cells),
                                 dir_prior=tuple(args.dir_prior),
                                 )

    # Save to output
    print('Saving synthetic ST simulation...')
    expr_path = args.expr_file.rpartition('/')[0]
    mix_path = args.mix_file.rpartition('/')[0]
    if not os.path.exists(expr_path):
        os.makedirs(expr_path)
    if not os.path.exists(mix_path):
        os.makedirs(mix_path)

    df_st.to_csv(args.expr_file, index=True)
    df_mix.to_csv(args.mix_file, index=True)
