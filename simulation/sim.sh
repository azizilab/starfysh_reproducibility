#!/usr/bin/sh

# Input file paths
# raw counts, barcodes, genes & metadata files of input scRNA-seq
rc="../data/wu_data/CID44971/scrna/bc_sc.csv" 
meta="../data/wu_data/CID44971/scrna/metadata.csv"
sig="data/tnbc_signature.csv"  # Signature gene sets for GSVA calculation

# Output file paths
# debug
exp="data/counts.st_synth.csv" # output ST expression
#gs_path="data_stereo/"
gs_path="data_stereo/"

# Input parameters
n_spots=$1
n_genes=$2

echo " Simulating synthetic ST from scRNA-seq dataset"
echo "============\n"
#./simulate.py --rc $rc --barcode $bc --gene $ge --meta $meta --expr $exp --mix $ct --transpose

./stereo_sim.py -c $rc -l $meta -ns $n_spots -ng $n_genes -o $gs_path

