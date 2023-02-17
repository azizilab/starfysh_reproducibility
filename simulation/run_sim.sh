#!/usr/bin/bash

n_ct=$1
mu=$2

cell_types=("Cancer Basal SC" "T_cells_c0_CD4+_CCR7" "T_cells_c7_CD8+_IFNG" "B cells Memory" "Myeloid_c9_Macrophage_2_CXCL10" "Myeloid_c4_DCs_pDC_IRF7" "CAFs MSC iCAF-like s1" "CAFs myCAF like s5" "Endothelial ACKR1" "PVL Immature s1")

#for elem in "${cell_types[@]}"
#do
#  echo $elem
#done


sc_path='../../starfysh/data/CID44971_TNBC/scrna/'
st_path='../../starfysh/data/CID44971_TNBC/'
sample_id='CID44971_TNBC'
outdir='../data/simu_defined/'

./spatial_sim.py \
  -r $sc_path \
  -i $sample_id \
  -s $st_path \
  --names "${cell_types[@]}" \
  -nc $n_ct \
  -mu $mu \
  -o $outdir
