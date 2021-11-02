#require(remotes)
#remotes::install_github('JEFworks-Lab/MERINGUE', build_vignettes = TRUE)
# normalize across bns

rm(list=ls())

suppressMessages(library(MERINGUE))
library(ComplexHeatmap)

setwd('~/Downloads/bc_visium_mod/')

all_bc_meta = read.csv('~/Downloads/bc_visium_mod/all_bc_meta.csv')

#lisa_df = data.frame()
slisa_df_all = data.frame()
source_all = c()
bn_df_all = data.frame()
ct_all = c()
slisa_all = data.frame()

for(i in seq(1,nrow(all_bc_meta))){
  sample_id = all_bc_meta[i,'sample_id']
  bn_df = read.csv(paste0('signature_v1_res/',all_bc_meta[i,'bn_file']),row.names = 1)
  pos = read.csv(all_bc_meta[i,'position_file'],row.names = 1,header = FALSE)
  
  bn_df_n = bn_df
  for(col in colnames(bn_df)){
    bn_df_n[,col] = bn_df[,col]/max(max(bn_df))
  }
  row.names(bn_df_n)= paste0(row.names(bn_df),'_',sample_id)
  row.names(pos)= paste0(row.names(pos),'_',sample_id)
  bn_df_all = rbind(bn_df_all,bn_df_n)
  source_all = c(source_all,rep(all_bc_meta[i,'source'],nrow(bn_df)))
  ct_all = c(ct_all,rep(all_bc_meta[i,'cancer_type'],nrow(bn_df)))      
  
  weight <- getSpatialNeighbors(pos[row.names(bn_df_n),c('V5','V6')])
  tmp = data.frame(row.names = row.names(bn_df_n))
  for(bn in colnames(bn_df)){
    gexp <- bn_df_n[,bn]#/max(bn_df_2A_n[,bn])#bn_all_n[,col] = bn_all[,col]/max(bn_all[,col])
    names(gexp) = row.names(bn_df_n)
    slisa <- signedLisa(gexp, weight)
    tmp[,bn] = slisa
  }
  
  slisa_all = rbind(slisa_all,tmp)
  
}

write.csv(bn_df_all,file='~/Downloads/bc_visium_mod/bn_all_tumors.csv')
write.csv(source_all,file='~/Downloads/bc_visium_mod/source_all_tumors.csv')
write.csv(ct_all,file='~/Downloads/bc_visium_mod/ct_all_tumors.csv')
write.csv(slisa_all,file='~/Downloads/bc_visium_mod/slisa_all_tumors.csv')
#####################################################################################

rm(list=ls())
suppressMessages(library(MERINGUE))
library(ComplexHeatmap)

bn_df_all=read.csv('~/Downloads/bc_visium_mod/bn_all_tumors.csv',row.names = 1)
source_all=read.csv('~/Downloads/bc_visium_mod/source_all_tumors.csv',row.names = 1)
ct_all=read.csv('~/Downloads/bc_visium_mod/ct_all_tumors.csv',row.names = 1)
slisa_all=read.csv('~/Downloads/bc_visium_mod/slisa_all_tumors.csv',row.names = 1)

cts = ct_all$x
setwd('~/Downloads/bc_visium_mod/')
meta_df = read.csv('~/Downloads/bc_visium_mod/BC_Data/metadata_n.csv')
#meta_df
patients = unique(meta_df[c("patient")])
replicates = unique(meta_df$replicate)

for(p in patients$patient){
  for(repl in replicates){
    
    bn_all = read.csv(paste0('signature_v1_res/',p,'_bn_df_all.csv'),row.names = 1)
    #row.names(bn_all)= paste0(row.names(bn_all),'_',p,'_',repl)
    
    #repl = strsplit(row.names(bn_all)[1],split = '_')[[1]][3]
    file = meta_df[(meta_df$patient==p)&(meta_df$replicate==repl),'spot_coordinates']
    if(length(file)>0){
      
      pos = read.csv(paste0('~/Downloads/bc_visium_mod/BC_Data/',strsplit(file,split = '[.]')[[1]][1],'.csv'),row.names = 1,header = TRUE)#read.csv(paste0('~/Downloads/bc_visium_mod/BC_Data/spots_',p,'_',repl,'.csv'),row.names = 1,header = FALSE)
      tt = row.names(pos)
      row.names(pos) = paste0(tt,'_',p,'_',repl)
      spots = intersect(row.names(pos),row.names(bn_all))
      
      tmp = data.frame(row.names = spots)
      
      weight <- getSpatialNeighbors(pos[spots,])
      
      I <- getSpatialPatterns(t(bn_all[spots,]), weight)
      
      for(bn in colnames(bn_all)){
        gexp <- bn_all[spots,bn]/max(max(bn_all[spots,]))#bn_all_n[,col] = bn_all[,col]/max(bn_all[,col])
        names(gexp) = spots#row.names(bn_all)
        slisa <- signedLisa(gexp, weight)
        tmp[,bn] = slisa
      }
      slisa_all = rbind(slisa_all,tmp)
      source_all = c(source_all,rep(p,times=length(spots)))
      
      cts = c(cts,rep(meta_df[(meta_df$patient==p)&(meta_df$replicate==repl),'type'],length(spots))) 
    }}}

bn_df_a_s = bn_df_all[names(which(apply(abs(bn_df_all), 1, max)>0.2)),]
slisa_df_a_s = slisa_all[names(which(apply(abs(slisa_all), 1, max)>1)),]

cts_df = data.frame(ct=cts)
row.names(cts_df) = row.names(slisa_all)

#write.csv(bn_df_a_s,file='~/Downloads/bc_visium_mod/bn_all_tumors_s.csv')
write.csv(slisa_df_a_s,file='~/Downloads/bc_visium_mod/slisa_all_all_tumors_s.csv')


slisa_biclust = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/slisa_alltumors_bicluster_res_new_s_all.csv',row.names = 1)
slisa_biclust[1:4,1:4]  

slisa_biclust_s = slisa_biclust[names(which(apply(abs(slisa_biclust), 1, max)>1)),]

row_cl = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/slisa_alltumors_bicluster_res_row_orders_new_s_all.csv')
head(row_cl)
row.names(row_cl) = row_cl$id
table(row_cl$cluster)
table(row_cl[row.names(slisa_biclust_s),'cluster'],cts_df[row.names(slisa_biclust_s),'ct'])


col_cl = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/slisa_alltumors_bicluster_res_col_orders_new_s_all.csv')
head(col_cl)
row.names(col_cl) = col_cl$id
table(col_cl$cluster)

bn_dict = data.frame(c('BC', 'bcell', 'CD8_T_cell_activation',
                       'CD8+_deletional_tolerance', 'CD8+_TIL_dysfunction',
                       'Effector_cell_cytotoxicity', 'Exhaustion_Terminal_differentiation',
                       'Fibroblast', 'GNF2_MKI67', 
                       'Macrophage', 'Monocyte', 'NK', 'Tfh', 'Th17', 'Th22', 'Th9', 'Treg'),
                     c(paste0('bn',seq(1,17))))

colnames(bn_dict) = c('cell_type','bn')

new_col = c()
for(i in colnames(slisa_biclust_s)){
  new_col = c(new_col,bn_dict[bn_dict$bn==i,'cell_type'])
}
colnames(slisa_biclust_s) = new_col
#colnames(bn_df_a_s) = new_col
# https://www.r-graph-gallery.com/ggplot2-color.html
row_cl$cluster = as.factor(row_cl$cluster)
cts_df$ct = as.factor(cts_df$ct)
col1 = structure(c('cyan2','darkgreen','darkgoldenrod','darkorange','darkolivegreen1','darksalmon','deepskyblue','chartreuse4','coral4','coral1','darkgoldenrod1'),names=c('ER+', 'ER+PR-Her2+', 'ER+PR-HER2+', 'ER+PR+HER2-','metaplastic','TNBC','HER2_luminal','HER2_non_luminal','Luminal_A','Luminal_B'))
h1 = Heatmap(slisa_biclust_s,show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,row_order = row.names(slisa_biclust_s),column_order = colnames(slisa_biclust_s),name = 'bn_slisa')
#h2 = Heatmap(sources_a[row.names(slisa_biclust_s),],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE)
#colors = structure(c('blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan'), names = c('ER+', 'ER+PR-Her2+', 'ER+PR-HER2+', 'ER+PR+HER2-','metaplastic','TNBC'))
h3 = Heatmap(cts_df[row.names(slisa_biclust_s),'ct'],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,name='cancer_type',col=col1)#,col=colors
h4 = Heatmap(row_cl[row.names(slisa_biclust_s),'cluster'],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,col=c('blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan'),name='hub')
#draw(h1+h2+h3+h4)
draw(h1+h3+h4)


############################ stacked bar plots

tt = as.data.frame.matrix(table(row_cl$cluster,cts_df[row.names(row_cl),'ct']))

library(ggplot2)
bar_plot_df = data.frame()
for(i in colnames(tt)){
  bar_plot_df = rbind(bar_plot_df,data.frame(ct=rep(i,times=nrow(tt)),hub=row.names(tt),val=tt[,i]))
}
#http://www.stat.columbia.edu/~tzheng/files/Rcolor.pdf
ggplot(bar_plot_df, aes(fill=hub, y=val, x=ct)) + 
  geom_bar(position="fill", stat="identity")+
  theme(text = element_text(size=20),
        axis.text.x = element_text(angle=90, hjust=1))+
  scale_fill_manual(values = c('blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan'))
# 'blue','orange','green','red','brown','pink','olive','cyan'
#'cyan2','darkgreen','darkgoldenrod','darkorange','darkolivegreen1','darksalmon','deepskyblue','chartreuse'
#'blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan'
### bar plot for bns

###### patterns generate edge files

for(ii in unique(row_cl$cluster)){
  cl_rows = row_cl[row_cl$cluster==ii,'id']
  node_df = data.frame()
  cts = unique(cts_df$ct)
  cts_df['id'] = row.names(cts_df)
  
  for(i in bn_dict$bn){
    for(j in cts){
      
      spots_ct = intersect(cl_rows,cts_df[cts_df$ct==j,'id'])
      node_df = rbind(node_df,data.frame(ct=j,node=bn_dict[bn_dict$bn==i,'cell_type'],slisa=sum(slisa_biclust_s[spots_ct,bn_dict[bn_dict$bn==i,'cell_type']])/length(spots_ct)))
    }}
  # edge df
  pairs = t(combn(bn_dict$bn,2))
  edge_df = data.frame()
  
  for(i in cts){
    spots_ct = intersect(cl_rows,cts_df[cts_df$ct==i,'id'])
    for(pa in seq(1,nrow(pairs))){
      p1 = bn_dict[bn_dict$bn==pairs[pa,1],'cell_type']
      p2 = bn_dict[bn_dict$bn==pairs[pa,2],'cell_type']
      edge_df = rbind(edge_df,data.frame(ct=i,node1=p1,node2=p2,cor=sum(slisa_biclust_s[spots_ct,p1]*slisa_biclust_s[spots_ct,p2])/length(spots_ct)))
      #edge_df = rbind(edge_df,data.frame(ct=i,node1=p1,node2=p2,cor=cor(slisa_biclust_s[spots_ct,p1],slisa_biclust_s[spots_ct,p2])))
    }
  }
  
  write.csv(node_df,file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_nodes_v2.csv'))
  write.csv(edge_df,file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_edges_v2.csv'))
  
  write.csv(na.omit(node_df[node_df$ct=='metaplastic',]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_nodes_metaplastic_v2.csv'))
  #write.csv(na.omit(edge_df[(edge_df$ct=='metaplastic')&(abs(edge_df$cor)>0.25),]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_edges_metaplastic_v2.csv'))
  write.csv(na.omit(edge_df[edge_df$ct=='metaplastic',]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_edges_metaplastic_v2.csv'))
  
  write.csv(na.omit(node_df[node_df$ct=='TNBC',]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_nodes_TNBC_v2.csv'))
  #write.csv(na.omit(edge_df[(edge_df$ct=='TNBC')&(abs(edge_df$cor)>0.25),]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_edges_TNBC.csv'))
  write.csv(na.omit(edge_df[edge_df$ct=='TNBC',]),file = paste0('~/Downloads/bc_visium_mod/slisa_biclust_cl',ii,'_edges_TNBC_v2.csv'))
  
} 

#################### v2
rm(list=ls())

suppressMessages(library(MERINGUE))
library(ComplexHeatmap)

setwd('~/Downloads/bc_visium_mod/')

all_bc_meta = read.csv('~/Downloads/bc_visium_mod/all_bc_meta.csv')

bn_df = read.csv('signature_v2_res/bn_df_1A_n.csv',row.names = 1)
pos = read.csv('~/Downloads/bc_visium_mod/data/MICHELLE_0212_AHJTWTDRXX__Project_10778__Sample_Patient1A_012020_IGO_10778_1/outs/spatial/tissue_positions_list.csv',row.names = 1,header = FALSE)
  
bn_df_n = bn_df
for(col in colnames(bn_df)){
    bn_df_n[,col] = bn_df[,col]/max(max(bn_df))
  }

weight <- getSpatialNeighbors(pos[row.names(bn_df_n),c('V5','V6')])
tmp = data.frame(row.names = row.names(bn_df_n))
for(bn in colnames(bn_df)){
    gexp <- bn_df_n[,bn]#/max(bn_df_2A_n[,bn])#bn_all_n[,col] = bn_all[,col]/max(bn_all[,col])
    names(gexp) = row.names(bn_df_n)
    slisa <- signedLisa(gexp, weight)
    tmp[,bn] = slisa
  }

library(ComplexHeatmap)  

bn_dict = data.frame(c('Basal_SC2', 'bcell', 'CD8_T_cell_activation',
                       'CD8+_deletional_tolerance', 'CD8+_TIL_dysfunction','cDC','dCAFs',
                       'Effector_cell_cytotoxicity','Endothelial', 'Exhaustion_Terminal_differentiation',
                        'GNF2_MKI67', 'Her2E_SC2','LumA_SC2','LumB_SC2',
                       'Macrophage', 'mCAFs', 'MBC','NK', 'pDC', 'Th17', 'Treg','vCAFs'),
                     c(paste0('bn',seq(1,22))))

colnames(bn_dict) = c('cell_type','bn')

new_col = c()
for(i in colnames(tmp)){
  new_col = c(new_col,bn_dict[bn_dict$bn==i,'cell_type'])
}
colnames(tmp) = new_col


Heatmap(tmp,show_row_names = FALSE)

library(CondIndTests)
lr_pairs = read.table('~/Documents/YeMac/headneck tcell data/PairsLigRec.txt',sep = '\t',header = TRUE)

# more samples
all_bc_meta = read.csv('~/Downloads/bc_visium_mod/all_bc_meta.csv')

#lisa_df = data.frame()
slisa_df_all = data.frame()
source_all = c()
bn_df_all = data.frame()
ct_all = c()
slisa_all = data.frame()

all_bc_meta_n = all_bc_meta[all_bc_meta$bn_file_v2!=0,]
for(i in seq(1,nrow(all_bc_meta_n))){
  sample_id = all_bc_meta_n[i,'sample_id']
  bn_df = read.csv(paste0('signature_v2_res/',all_bc_meta_n[i,'bn_file_v2']),row.names = 1)
  pos = read.csv(all_bc_meta_n[i,'position_file'],row.names = 1,header = FALSE)
  
  bn_df_n = bn_df
  for(col in colnames(bn_df)){
    bn_df_n[,col] = bn_df[,col]/max(max(bn_df))
  }
  row.names(bn_df_n)= paste0(row.names(bn_df),'_',sample_id)
  row.names(pos)= paste0(row.names(pos),'_',sample_id)
  bn_df_all = rbind(bn_df_all,bn_df_n)
  source_all = c(source_all,rep(all_bc_meta_n[i,'source'],nrow(bn_df)))
  ct_all = c(ct_all,rep(all_bc_meta_n[i,'cancer_type'],nrow(bn_df)))      
  
  weight <- getSpatialNeighbors(pos[row.names(bn_df_n),c('V5','V6')])
  tmp = data.frame(row.names = row.names(bn_df_n))
  for(bn in colnames(bn_df)){
    gexp <- bn_df_n[,bn]#/max(bn_df_2A_n[,bn])#bn_all_n[,col] = bn_all[,col]/max(bn_all[,col])
    names(gexp) = row.names(bn_df_n)
    slisa <- signedLisa(gexp, weight)
    tmp[,bn] = slisa
  }
  
  slisa_all = rbind(slisa_all,tmp)
  
}

write.csv(bn_df_all,file='~/Downloads/bc_visium_mod/signature_v2_res/bn_all_tumors.csv')
write.csv(source_all,file='~/Downloads/bc_visium_mod/signature_v2_res/source_all_tumors.csv')
write.csv(ct_all,file='~/Downloads/bc_visium_mod/signature_v2_res/ct_all_tumors.csv')
write.csv(slisa_all,file='~/Downloads/bc_visium_mod/signature_v2_res/slisa_all_tumors.csv')

rm(list=ls())
library(ComplexHeatmap)

slisa_all = read.csv('~/Downloads/bc_visium_mod/signature_v2_res/slisa_all_tumors.csv',row.names = 1)
bn_df_all = read.csv('~/Downloads/bc_visium_mod/signature_v2_res/bn_all_tumors.csv',row.names = 1)
source_all=read.csv('~/Downloads/bc_visium_mod/signature_v2_res/source_all_tumors.csv')
row.names(source_all) = row.names(bn_df_all)
slisa_biclust = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/signature_v2_res/slisa_alltumors_bicluster_res.csv',row.names = 1)
slisa_biclust[1:4,1:4]  

slisa_biclust_s = slisa_biclust[names(which(apply(abs(slisa_biclust), 1, max)>1)),]

row_cl = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/signature_v2_res/slisa_alltumors_bicluster_res_row_orders.csv')
head(row_cl)
row.names(row_cl) = row_cl$id
table(row_cl$cluster)


col_cl = read.csv('/Users/xueerchen/Downloads/bc_visium_mod/signature_v2_res/slisa_alltumors_bicluster_res_col_orders.csv')
head(col_cl)
row.names(col_cl) = col_cl$id
table(col_cl$cluster)

bn_dict = data.frame(c('Basal_SC2', 'bcell', 'CD8_T_cell_activation',
                       'CD8+_deletional_tolerance', 'CD8+_TIL_dysfunction','cDC','dCAFs',
                       'Effector_cell_cytotoxicity', 'Endothelial','Exhaustion_Terminal_differentiation','GNF2_MKI67',
                      'HER2E_SC2','LumA_sc2','LumB_sc2',
                       'Macrophage','mCAFs', 'MBC','NK','pDC', 'Th17', 'Treg',  'vCAFs'),
                     c(paste0('bn',seq(1,22))))

colnames(bn_dict) = c('cell_type','bn')

new_col = c()
for(i in colnames(slisa_biclust_s)){
  new_col = c(new_col,bn_dict[bn_dict$bn==i,'cell_type'])
}
colnames(slisa_biclust_s) = new_col
#colnames(bn_df_a_s) = new_col
# https://www.r-graph-gallery.com/ggplot2-color.html
row_cl$cluster = as.factor(row_cl$cluster)

cts_df = read.csv('~/Downloads/bc_visium_mod/signature_v2_res/ct_all_tumors.csv',row.names = 1)
row.names(cts_df) = row.names(slisa_all)
cts_df$x = as.factor(cts_df$x)
col1 = structure(c('cyan2','darkgreen','darkgoldenrod','darkorange','darkolivegreen1','darksalmon','deepskyblue','chartreuse4','coral4','coral1','darkgoldenrod1'),names=c('ER+', 'ER+PR-Her2+', 'ER+PR-HER2+', 'ER+PR+HER2-','metaplastic','TNBC','HER2_luminal','HER2_non_luminal','luminalA','luminalB'))
h1 = Heatmap(slisa_biclust_s,show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,row_order = row.names(slisa_biclust_s),column_order = colnames(slisa_biclust_s),name = 'bn_slisa')
#h2 = Heatmap(sources_a[row.names(slisa_biclust_s),],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE)
#colors = structure(c('blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan'), names = c('ER+', 'ER+PR-Her2+', 'ER+PR-HER2+', 'ER+PR+HER2-','metaplastic','TNBC'))
h3 = Heatmap(cts_df[row.names(slisa_biclust_s),'x'],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,name='cancer_type',col=col1)#,col=colors
h4 = Heatmap(row_cl[row.names(slisa_biclust_s),'cluster'],show_row_names = FALSE,row_dend_reorder = FALSE,column_dend_reorder = FALSE,col=c('blue','orange','darkolivegreen1','red','brown','pink','darkolivegreen4','cyan','coral1','darkgoldenrod1','green'),name='hub')
#draw(h1+h2+h3+h4)
draw(h1+h3+h4)
#draw(h1+h3)


bn_dict = data.frame(c('Basal_SC2', 'bcell', 'CD8_T_cell_activation',
                       'CD8+_deletional_tolerance', 'CD8+_TIL_dysfunction','cDC','dCAFs',
                       'Effector_cell_cytotoxicity', 'Endothelial','Exhaustion_Terminal_differentiation','GNF2_MKI67',
                       'HER2E_SC2','LumA_sc2','LumB_sc2',
                       'Macrophage','mCAFs', 'MBC','NK','pDC', 'Th17', 'Treg',  'vCAFs'),
                     c(paste0('bn',seq(1,22))))

colnames(bn_dict) = c('cell_type','bn')

new_col = c()
for(i in colnames(bn_df_all)){
  new_col = c(new_col,bn_dict[bn_dict$bn==i,'cell_type'])
}
colnames(bn_df_all) = new_col

library(ppcor)

spots_hub7 = row.names(row_cl[row_cl$cluster=='7',])
#bn_df_all[spots_hub7,]

lr_pair_df = read.table('~/Documents/YeMac/headneck tcell data/PairsLigRec.txt',sep = '\t',header = TRUE)

gexp1 = read.csv('~/Downloads/bc_visium_mod/signature_v1_res/gexp_1A.csv',row.names = 1)
row.names(gexp1) = paste0(row.names(gexp1),'_1A')
gexp2 = read.csv('~/Downloads/bc_visium_mod/signature_v1_res/gexp_1B.csv',row.names = 1)
row.names(gexp2) = paste0(row.names(gexp2),'_1B')

gexp = rbind(gexp1,gexp2)

bn_pairs = combn(c('GNF2_MKI67','CD8+_deletional_tolerance','bcell','CD8+_TIL_dysfunction','Th17', 'dCAFs','mCAFs','pDC'),2)

spots_hub7 = intersect(spots_hub7,row.names(gexp))
for(ii in seq(1,ncol(bn_pairs))){
  cors = c()
  pvs = c()
  lrs = c()
  for(i in seq(1,nrow(lr_pair_df))){
    tryCatch({
      #tmp = pcor.test(bn_df_1a$bn1,bn_df_1a$bn7,gexp[,c(lr_pair_df[i,'Ligand.ApprovedSymbol'],lr_pair_df[i,'Receptor.ApprovedSymbol'])])
      tmp = pcor.test(bn_df_all[spots_hub7,bn_pairs[1,ii]],bn_df_all[spots_hub7,bn_pairs[2,ii]],c(gexp[spots_hub7,lr_pair_df[i,'Ligand.ApprovedSymbol']],gexp[spots_hub7,lr_pair_df[i,'Receptor.ApprovedSymbol']]))
      
      cors = c(cors,tmp$estimate)
      pvs = c(pvs,tmp$p.value)
      lrs = c(lrs,lr_pair_df[i,'Pair.Name'])
    },
    error=function(e){})
    
  }
  pcor_res = data.frame(cor=cors,pv=pvs,lr=lrs)
  write.csv(pcor_res,file=paste0('~/Downloads/bc_visium_mod/signature_v2_res/lr_bn_ana_',bn_pairs[1,ii],'_',bn_pairs[2,ii],'.csv'))
  
}

lr_exp = matrix(,nrow=length(spots_hub7),ncol=nrow(lr_pair_df))#data.frame(row.names =spots_hub7)#,colnames=lr_pair_df$Pair.Name)#(nrow=nrow(gexp),ncol=nrow(lr_pair_df))
ttt=c()
for(i in seq(1,nrow(lr_pair_df))){
  if((lr_pair_df[i,'Ligand.ApprovedSymbol']%in%colnames(gexp))&(lr_pair_df[i,'Receptor.ApprovedSymbol']%in%colnames(gexp))){
    lr_exp[,i] = gexp[spots_hub7,lr_pair_df[i,'Ligand.ApprovedSymbol']]*gexp[spots_hub7,lr_pair_df[i,'Receptor.ApprovedSymbol']]
    ttt = c(ttt,lr_pair_df[i,'Pair.Name'])
  }
  #cbind(lr_exp,gexp[spots_hub7,lr_pair_df[i,'Ligand.ApprovedSymbol']]*gexp[spots_hub7,lr_pair_df[i,'Receptor.ApprovedSymbol']])
  
    #lr_exp[,lr_pair_df[i,'Pair.Name']] = t(gexp[spots_hub7,lr_pair_df[i,'Ligand.ApprovedSymbol']]*gexp[spots_hub7,lr_pair_df[i,'Receptor.ApprovedSymbol']])
  }
lr_exp[1:4,1:4] 
  
### all spots

n1 = c()
n2 = c()
cor_c = c()
lr = c()
for(ii in seq(1,ncol(bn_pairs))){
  cors = c()
  pvs = c()
  lrs = c()
  for(i in seq(1,nrow(lr_pair_df))){
    tryCatch({
      #tmp = pcor.test(bn_df_1a$bn1,bn_df_1a$bn7,gexp[,c(lr_pair_df[i,'Ligand.ApprovedSymbol'],lr_pair_df[i,'Receptor.ApprovedSymbol'])])
      tmp = pcor.test(bn_df_all[row.names(gexp),bn_pairs[1,ii]],bn_df_all[row.names(gexp),bn_pairs[2,ii]],c(gexp[,lr_pair_df[i,'Ligand.ApprovedSymbol']],gexp[,lr_pair_df[i,'Receptor.ApprovedSymbol']]))
      
      cors = c(cors,tmp$estimate)
      pvs = c(pvs,tmp$p.value)
      lrs = c(lrs,lr_pair_df[i,'Pair.Name'])
    },
    error=function(e){})
    
  }
  pcor_res = data.frame(cor=cors,pv=pvs,lr=lrs)
  write.csv(pcor_res,file=paste0('~/Downloads/bc_visium_mod/signature_v2_res/lr_bn_ana_',bn_pairs[1,ii],'_',bn_pairs[2,ii],'.csv'))
  n1 = c(n1,bn_pairs[1,ii])
  n2 = c(n2,bn_pairs[2,ii])
  cor_c = c(cor_c,pcor_res[order(pcor_res$cor,decreasing = TRUE),'cor'][1])
  lr = c(lr,pcor_res[order(pcor_res$cor,decreasing = TRUE),'lr'][1])
}

cyto_df = data.frame('n1'=n1,'n2'=n2,'cor'=cor_c,'lr' = lr)
write.csv(cyto_df,file='~/Downloads/bc_visium_mod/signature_v2_res/cyto_df.csv')
