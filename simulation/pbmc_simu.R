##### x
rm(list=ls())
library(GSVA)
exp_df = read.csv('~/Downloads/bc_visium_mod/exp_markers_df_gsva_pbmc_x.csv',row.names = 1)
exp_df[1:4,1:4]

sigfile = read.csv('~/Downloads/bc_visium_mod/pbmc_markers_gsva_mod.csv',header=T)#read.csv('~/Downloads/bc_visium_mod/adA_gsva_markers_pbmc.csv',header=T)

sigfile$Cell.type = as.character(sigfile$Cell.type)
sigfile$Symbol = as.character(sigfile$Symbol)

table(sigfile$Cell.type)

genelist = split(as.character(sigfile[,2]),as.character(sigfile[,1]))

gsva_scores = gsva(t(exp_df),genelist,method="ssgsea")
heatmap(gsva_scores)
gsva_scores = gsva(t(exp_df),genelist,method="gsva")
heatmap(gsva_scores)

write.csv(t(gsva_scores),file='~/Downloads/bc_visium_mod/markers_gsva_pbmc_x.csv')
#write.csv(gsva_scores,file='~/Downloads/bc_visium_mod/markers_gsva_pbmc_quartz.csv')

