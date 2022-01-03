rm(list=ls())

sample_id = '1A'#all_bc_meta_n[i,'sample_id']
bn_df = read.csv('signature_v3_res/bn_df_1A_individual.csv',row.names = 1)
pos = read.csv('data/MICHELLE_0212_AHJTWTDRXX__Project_10778__Sample_Patient1A_012020_IGO_10778_1/outs/spatial/tissue_positions_list.csv',row.names = 1,header = FALSE)
row.names(pos)= paste0(row.names(pos),'_',sample_id)

bn_dict = data.frame(c('bcell', 'CAFs_MSC_iCAF-like', 'CAFs_myCAF-like',
                       'CD8_T_cell_activation', 'CD8+_deletional_tolerance',
                       'CD8+_TIL_dysfunction', 'cDC', 'dCAFs', 'Endothelial', 'Macrophage_M1',
                       'Macrophage_M2', 'MBC', 'mCAFs', 'MDSC', 'NK', 'Normal_epithelial',
                       'pDC', 'Plasmablasts', 'Precursor_exhaustion', 'PVL_Differentiated',
                       'PVL_Immature', 'Terminal_exhaustion', 'Th17', 'TNBC', 'Treg_wu',
                       'vCAFs'),
                     c(paste0('bn',seq(1,26))))

colnames(bn_dict) = c('cell_type','bn')


bn_df_n = bn_df
for(col in colnames(bn_df)){
  bn_df_n[,col] = (bn_df[,col] - mean(bn_df[,col]))/sd(bn_df[,col])#bn_df[,col]/max(max(bn_df))
}
#row.names(bn_df_n)= paste0(row.names(bn_df),'_',sample_id)

new_col = c()
for(i in colnames(bn_df)){
  new_col = c(new_col,bn_dict[bn_dict$bn==i,'cell_type'])
}
colnames(bn_df) = new_col

weight <- getSpatialNeighbors(pos[row.names(bn_df),c('V5','V6')])

scc_score_bn = spatialCrossCorMatrix(t(bn_df),weight)

lr_pairs = read.table('~/Documents/YeMac/headneck tcell data/PairsLigRec.txt',sep = '\t',header = TRUE)
lr_exp = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/signature_v1_res/gexp_1A.csv',row.names = 1)
tmp = row.names(lr_exp)
row.names(lr_exp) = paste0(tmp,'_1A')

# use scc_score_bn to filter bn pairs
bn_pairs = combn(colnames(bn_df),2)

# binarize bn df to select pure spots, position

bn_df_bi = bn_df
for(i in colnames(bn_df_bi)){
  tmp = as.numeric(quantile(bn_df[,i],probs=seq(0,1,0.15))['75%'])
  bn_df_bi[,i] = as.integer(bn_df[,i]>tmp)
}

f1s = c()
f2s = c()
iscc = c()
lis = c()
res = c()

for(j in seq(1,ncol(bn_pairs))){
  
  if(scc_score_bn[bn_pairs[1,j],bn_pairs[2,j]]>0.1){
    for(i in seq(1,nrow(lr_pairs))){
      if((lr_pairs[i,'Ligand.ApprovedSymbol'] %in% colnames(lr_exp))&(lr_pairs[i,'Receptor.ApprovedSymbol'] %in% colnames(lr_exp))){
        
        groupA = row.names(bn_df_bi)[which(bn_df_bi[,bn_pairs[1,j]]>0)]
        groupB = row.names(bn_df_bi)[which(bn_df_bi[,bn_pairs[2,j]]>0)]
        #interscc = interCellTypeSpatialCrossCor(gexpA, gexpB, groupA, groupB, weight) 
        
        weightIc <- getInterCellTypeWeight(groupA, groupB, 
                                           weight, pos[,c('V5','V6')], 
                                           #plot=TRUE, 
                                           main='Adjacency Weight Matrix\nBetween Cell-Types')
        #spatialCrossCor(gexp3, gexp4, weightIc)
        
        gexpA = lr_exp[row.names(weightIc),lr_pairs[i,'Ligand.ApprovedSymbol']]
        gexpB = lr_exp[row.names(weightIc),lr_pairs[i,'Receptor.ApprovedSymbol']]
        
        names(gexpA) = row.names(weightIc)
        names(gexpB) = row.names(weightIc)
        
        tt = interCellTypeSpatialCrossCor(gexpA, gexpB, groupA, groupB, weightIc)
        
        f1s = c(f1s,bn_pairs[1,j])
        f2s = c(f2s,bn_pairs[2,j])
        iscc = c(iscc,tt)
        lis = c(lis,lr_pairs[i,'Ligand.ApprovedSymbol'])
        res = c(res,lr_pairs[i,'Receptor.ApprovedSymbol'])
        
      }
      
    }}}

iscc_lr_res = data.frame(f1 = f1s,
                         f2 = f2s,
                         score = iscc,
                         ligand= lis,
                         receptor = res)

write.csv(iscc_lr_res,file='~/Downloads/bc_visium_mod/signature_v3_res/iscc_lr_res_1A.csv')
