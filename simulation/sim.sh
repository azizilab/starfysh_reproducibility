#!/usr/bin/sh

# Input file paths
# raw counts, barcodes, genes & metadata files of input scRNA-seq
rc="../data/wu_data/CID44971/scrna/count_matrix_sparse.mtx" 
bc="../data/wu_data/CID44971/scrna/count_matrix_barcodes.tsv"
ge="../data/wu_data/CID44971/scrna/count_matrix_genes.tsv"
meta="../data/wu_data/CID44971/scrna/metadata.csv"

sig="data/tnbc_signature.csv"  # Signature gene sets for GSVA calculation

# Output file paths
exp="data/bc_spot_exp.csv" # output ST expression
ct="data/bc_spot_ct.csv" # output ST cell counts
gs_path="data/"
gs="marker_gsva_bc.csv" # output ST GSVA scores

echo "[Step 1]: Simulating synthetic ST from scRNA-seq dataset"
echo "============\n"
./simulate.py --rc $rc --barcode $bc --gene $ge --meta $meta --expr $exp --mix $ct --transpose

echo "[Step 2]: Calculate GSVA from synthesized ST data"
echo "============\n"
./gsva.R --exp $exp --sig $sig -o $gs_path --name $gs

