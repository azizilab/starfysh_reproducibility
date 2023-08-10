#!/usr/bin/bash

n_ct=$1
mu=$2

# 5 major cell type simulation
cell_types=("CAFs" "Cancer Epithelial" "Myeloid" "Normal Epithelial" "T-cells")
ct="major"

# 10 fine-grained cell type simulation
#cell_types=("Cancer Basal SC" "T_cells_c0_CD4+_CCR7" "T_cells_c7_CD8+_IFNG" "B cells Memory" "Myeloid_c4_DCs_pDC_IRF7" "T_cells_c2_CD4+_T-regs_FOXP3" "CAFs MSC iCAF-like s1" "CAFs myCAF like s5" "Endothelial ACKR1" "PVL Immature s1")
#ct="subset"

sc_path='../../starfysh/data/CID44971_TNBC/scrna/'
st_path='../../starfysh/data/CID44971_TNBC/'
sample_id='CID44971_TNBC'
#outdir='../data/simu_5/'
#outdir='../data/simu_10/'
outdir='../data/simu_debug/'

./spatial_sim.py \
  -r $sc_path \
  -i $sample_id \
  -s $st_path \
  --names "${cell_types[@]}" \
  -ct $ct \
  -nc $n_ct \
  -mu $mu \
  -o $outdir
