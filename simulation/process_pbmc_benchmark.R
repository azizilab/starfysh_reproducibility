rm(list=ls())
#library("devtools")
#install_github("elimereu/matchSCore2")
# https://github.com/elimereu/matchSCore2/blob/master/README.md
#if (!requireNamespace("BiocManager", quietly = TRUE))
  #install.packages("BiocManager")
#BiocManager::install("scater")
library(scater)
load('~/Documents/YeMac/visium_bc/sce.all_classified.technologies.RData')
colData(sce)
table(colData(sce)$batch) ## Number of cells from each protocol
table(colData(sce)$nnet2) ## Number of cells from each classified cell type (by matchSCore2 classification)
table(colData(sce)$ident) ## Number of cells from each Seurat cluster

#quartzseq2 <- scater::filter(sce,batch=="Quartz-Seq2") ### new object with cells from Quartz-Seq2 protocol.
#counts <- quartzseq2@assays$data@listData$counts ## Quartz-Seq2 UMI counts
#logcounts <- quartzseq2@assays$data@listData$logcounts ## Quartz-Seq2 logcounts
#meta.data <- as.data.frame(colData(quartzseq2)) ## Quartz-Seq2 meta.data dataframe

sce_quartz <- sce[, sce$batch == "Quartz-Seq2"]
sce_quartz$nnet2 <- as.character(sce_quartz$nnet2)
table(sce_quartz$batch)
table(sce_quartz$nnet2)

meta_df = data.frame(sce_quartz@colData)
exp_df = sce_quartz@assays$data$counts
unique(meta_df$nnet2)

write.csv(t(exp_df),'~/Downloads/ST_simulation-master/pbmc/quartz_exp.csv')
write.csv(meta_df,'~/Downloads/ST_simulation-master/pbmc/quartz_meta.csv')

##### smartseq
sce_Smart <- sce[, sce$batch == "Smart-Seq2"]
sce_Smart$nnet2 <- as.character(sce_Smart$nnet2)
table(sce_Smart$batch)
table(sce_Smart$nnet2)

meta_df = data.frame(sce_Smart@colData)
exp_df = sce_Smart@assays$data$counts
unique(meta_df$nnet2)

write.csv(t(exp_df),'~/Downloads/ST_simulation-master/pbmc/Smart-Seq2_exp.csv')
write.csv(meta_df,'~/Downloads/ST_simulation-master/pbmc/Smart-Seq2_meta.csv')



#table(colData(sce)$ident)
#sce_quartz <- sce_quartz[, sce_quartz$nnet2 != "Megakaryocytes"]

#se_quartz <- Seurat::CreateSeuratObject(counts = sce_quartz@assays$data$counts,
                                        #meta.data = data.frame(sce_quartz@colDa#ta))
#saveRDS(object = se_quartz,
#        file = here::here(sprintf("%s/%s/se_quartz.RDS", an_tools, robj_dir)))

#synthetic_mixtures <- SPOTlight::test_spot_fun(se_obj = se_quartz,
                                               #clust_vr = "nnet2",
                                               #n = 1000,
                                               #verbose = TRUE)
#saveRDS(object = synthetic_mixtures,
       # file = here::here(sprintf("%s/%s/common_synthetic_mixtures.RDS", an_tools, robj_dir)))

# random sampling

# 2&8 cells per spot, keep all the genes, 6000 spots
# random sample from unique cell types, and random numbers of cells, add togther
# generate multiple runnings
num_cells = seq(2,8)
celltypes = unique(meta_df$nnet2)
num_spots = 6000

tt = seq(1,num_spots)
spot_exp_df = data.frame(row.names=paste0('s',tt)) #rbind each sample row data.frame()#
spot_ct_df = data.frame(row.names=paste0('s',tt)) # number of columns are number of unique cell types, each row represent one spot, entries are number of cells belong to that cell type


for(s in seq(1,num_spots)){
  num_cell = sample(num_cells,1)
  #celltype = sample(celltypes,size=num_cell,replace = TRUE)
  cell_ids = sample(colnames(exp_df),size=num_cell)
  spot_exp_df = rbind(spot_exp_df,as.numeric(apply(exp_df[,cell_ids],1,sum)))
  #spot_exp_df[s,] = as.numeric(apply(exp_df[,cell_ids],1,sum))#rbind(spot_exp_df,apply(exp_df[,cell_ids],1,sum))
  tmp = table(meta_df[cell_ids,'nnet2'])
  spot_ct_df = rbind(spot_ct_df,tmp[celltypes])# table(meta_df[cell_ids,'nnet2'])
  
}

#spot_exp_df = data.frame(spot_exp_df)
tt = seq(1,num_spots)
 
colnames(spot_exp_df) = row.names(exp_df)
row.names(spot_exp_df) = paste0('s',tt)
  
colnames(spot_ct_df) = celltypes
row.names(spot_ct_df) = paste0('s',tt)

spot_exp_df[1:4,1:4]
spot_ct_df[1:4,1:4]

write.csv(spot_exp_df,file='~/Documents/visium_bc/pbmc_benchmark_spot_exp_df.csv')
write.csv(spot_ct_df,file='~/Documents/visium_bc/pbmc_benchmark_spot_ct_df.csv')


####also generate cell type file

rm(list=ls())
exp_df = read.csv('~/Documents/visium_bc/pbmc_benchmark_spot_exp_df.csv',stringsAsFactors = FALSE,row.names = 1)

ct_df = read.csv('~/Documents/visium_bc/pbmc_benchmark_spot_ct_df.csv',stringsAsFactors = FALSE,row.names = 1)
tmp = rowSums(is.na(ct_df))

ae_bn = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse.csv',stringsAsFactors = FALSE,row.names = 1)
ae_bn[1:4,1:4]

library(ComplexHeatmap)
Heatmap(ae_bn,show_row_names = FALSE)

library(ConsensusClusterPlus)

##### all tps
title = 'pbmc_visium_bn_ae'
res = ConsensusClusterPlus(t(as.matrix(ae_bn)),maxK=25,reps=10,pItem=0.8,pFeature = 1,title = title,clusterAlg = 'km',distance ='euclidean',seed = 1262118388.71279,verbose=TRUE,plot='png')
#save(res,file='~/Documents/bc_visium_ae_bn_consensus_cluster_sparse.RData')

cl_df <- cbind(names(res[[9]]$consensusClass),res[[9]]$consensusClass-1)
#
colnames(cl_df) = c('spot_id','cluster')

bn_scores_cl = merge(ae_bn,cl_df,by='row.names')
bn_scores_cl$Row.names = NULL
#tumor_scores_cl$tumor_id = NULL
#
bn_scores_cl_s = bn_scores_cl[order(bn_scores_cl$cluster),]
#
bn_cls = bn_scores_cl_s$cluster
bn_scores_cl_s$cluster = NULL

bn_scores_cl_s$spot_id = NULL



h1= Heatmap(bn_scores_cl_s,cluster_rows = FALSE,cluster_columns = FALSE,width=6,column_names_gp = gpar(fontsize=7),column_dend_reorder = TRUE,show_column_names = TRUE,show_row_names = FALSE,show_column_dend = FALSE,show_heatmap_legend = FALSE,row_split = as.integer(bn_cls))#,column_title='SGAs' ,column_title='immuneGEM_enrichment_TCGA',column_title_gp = gpar(fontsize = 10, fontface = "bold")
h2 = Heatmap(as.integer(bn_cls),show_row_names = FALSE,show_column_names = FALSE,width = 0.1,col=structure(c("#A62A2A", "#FF1493", "#FF0000", "#FFD39B", "#FF7F24", "#CAFF70", "#228B22", "#20B2AA", "#00FF00", "#FFF68F", "#9932CC", "#98F5FF", "#1E90FF", "#0000FF", "#635688"),names=seq(1,14)),show_heatmap_legend = TRUE,heatmap_legend_param = list(title_gp = gpar(fontsize = 5),labels_gp=gpar(fontsize=5)))#col=structure(c("#FF7F24","#20B2AA","#A62A2A","#1E90FF"),names=c(1,2,3,4)),
pdf(file='~/Documents/pbmc_visium_bn_heatmap_cl_sparse.pdf')#, width = 10, height = 40)#,pointsize = 25)
draw(h1+h2)#, heatmap_legend_side = "bottom"
dev.off()

ct_df[is.na(ct_df)] <- 0

table(ct_df[,'HEK.cells'],cl_df[row.names(ct_df),'cluster'])

# draw ct heatmap
ct_cl = merge(ct_df,cl_df,by='row.names')
ct_cl$Row.names = NULL
#tumor_scores_cl$tumor_id = NULL
#
ct_cl_s = ct_cl[order(ct_cl$cluster),]
#
ct_cls = ct_cl_s$cluster
ct_cl_s$cluster = NULL

ct_cl_s$spot_id = NULL
ct_cl_s$cluster = NULL

ct_cl_s_n = ct_cl_s/rowSums(ct_cl_s)

h1= Heatmap(ct_cl_s_n,cluster_rows = FALSE,cluster_columns = FALSE,width=6,column_names_gp = gpar(fontsize=7),column_dend_reorder = TRUE,show_column_names = TRUE,show_row_names = FALSE,show_column_dend = FALSE,show_heatmap_legend = FALSE,row_split = as.integer(bn_cls))#,column_title='SGAs' ,column_title='immuneGEM_enrichment_TCGA',column_title_gp = gpar(fontsize = 10, fontface = "bold")
h2 = Heatmap(as.integer(ct_cls),show_row_names = FALSE,show_column_names = FALSE,width = 0.1,col=structure(c("#A62A2A", "#FF1493", "#FF0000", "#FFD39B", "#FF7F24", "#CAFF70", "#228B22", "#20B2AA", "#00FF00", "#FFF68F", "#9932CC", "#98F5FF", "#1E90FF", "#0000FF", "#635688"),names=seq(1,14)),show_heatmap_legend = TRUE,heatmap_legend_param = list(title_gp = gpar(fontsize = 5),labels_gp=gpar(fontsize=5)))#col=structure(c("#FF7F24","#20B2AA","#A62A2A","#1E90FF"),names=c(1,2,3,4)),
pdf(file='~/Documents/pbmc_visium_ct_heatmap_cl_sparse.pdf')#, width = 10, height = 40)#,pointsize = 25)
draw(h1+h2)#, heatmap_legend_side = "bottom"
dev.off()

captum_res = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse_captum.csv',stringsAsFactors = FALSE,row.names = 1)
captum_res[1:4,1:4]

max(captum_res)
min(captum_res)

vars = apply(captum_res, 2, var)
hist(vars)

# less input
captum_res_less = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse_captum_6000.csv',stringsAsFactors = FALSE,row.names = 1)
captum_res_less[1:4,1:4]

max(captum_res_less)
min(captum_res_less)

vars = apply(captum_res_less, 2, var)
hist(vars)

Heatmap(captum_res_less,show_row_dend = FALSE,show_column_dend = FALSE,show_row_names = TRUE,show_column_names = FALSE)

Heatmap(captum_res,show_row_dend = FALSE,show_column_dend = FALSE,show_row_names = FALSE,show_column_names = FALSE)

pbmc_markers = read.csv('~/Documents/visium_bc/pbmc_markers.csv',stringsAsFactors = FALSE)
head(pbmc_markers)

genes = c()
for(i in colnames(pbmc_markers)){
  genes = c(genes,pbmc_markers[,i])
}
genes = unique(genes)
length(genes)
length(intersect(genes,colnames(captum_res_less)))
length(intersect(genes,colnames(captum_res)))

Heatmap(captum_res[,intersect(genes,colnames(captum_res))],show_row_dend = FALSE,show_column_dend = FALSE,show_row_names = TRUE,show_column_names = FALSE)

rowSums(abs(captum_res_less))
Heatmap(captum_res_less[which(rowSums(abs(captum_res_less))>0),intersect(genes,colnames(captum_res_less))],show_row_dend = FALSE,show_column_dend = FALSE,show_row_names = TRUE,show_column_names = FALSE)

ae_bn = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse_6000.csv',stringsAsFactors = FALSE,row.names = 1)
ae_bn[1:4,1:4]

Heatmap(ae_bn[,which(colSums(abs(ae_bn))>0)],show_row_names = FALSE)

bn_celltype_df = data.frame()
for(i in colnames(pbmc_markers)){
  gene = pbmc_markers[,i]
  bn_celltype_df = rbind(bn_celltype_df,apply(captum_res_less[,intersect(gene,colnames(captum_res_less))],1,sum))
}

row.names(bn_celltype_df) = colnames(pbmc_markers)
colnames(bn_celltype_df) = paste0('bn',seq(1,20))

Heatmap(bn_celltype_df[,which(colSums(abs(bn_celltype_df))>0)])
####### maybe try to make prediction models
ct_df[is.na(ct_df)] <- 0

marker_captum_df = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse_captum_cellmarkers.csv',stringsAsFactors = FALSE,row.names = 1)
Heatmap(marker_captum_df,show_column_names = FALSE)

Heatmap(marker_captum_df[,intersect(genes,colnames(marker_captum_df))],show_column_names = FALSE)

for(i in colnames(pbmc_markers)){
  gene = pbmc_markers[,i]
  Heatmap(marker_captum_df[,intersect(gene,colnames(marker_captum_df))],show_column_names = FALSE)
}

bn_celltype_df = data.frame()
for(i in colnames(pbmc_markers)){
  gene = pbmc_markers[,i]
  bn_celltype_df = rbind(bn_celltype_df,apply(marker_captum_df[,intersect(gene,colnames(marker_captum_df))],1,sum))
}

row.names(bn_celltype_df) = colnames(pbmc_markers)
colnames(bn_celltype_df) = paste0('bn',seq(1,20))

Heatmap(bn_celltype_df[,which(colSums(abs(bn_celltype_df))>0)])

#library(mlr)

# only binary for multiabel

#multidata = cbind(ae_bn[,which(colSums(ae_bn)>0)],as.binary(ct_df>0))
#task = makeMultilabelTask(id='multi',data=multidata,target = colnames(ct_df))
library(xgboost)
library(ROCR)
library(caret)
library(dplyr)
library(stringr)

# normalize ct_df, row sum to be 1
ct_df_norm = ct_df/rowSums(ct_df)

set.seed(1234) # fixed results
splitIndex <- createDataPartition(y,p=0.75,list=FALSE,times=1)
x_train <- x[splitIndex,]
x_test <- x[-splitIndex,]
y_train <- y[splitIndex]
y_test <- y[-splitIndex]

train_matrix = xgb.DMatrix(data=x_train,label=y_train)
test_matrix = xgb.DMatrix(data=x_test,label=y_test)
v_matrix = xgb.DMatrix(data=x_v,label=y_v)
num_classes = 3
xgb_params = list("objective"="multi:softprob","eval_metric"="merror","num_class"=num_classes)#,"eval_metric"="logloss","num_class"=num_classes

nround = 50
cv.nfold = 5

cv_model = xgb.cv(params=xgb_params,data=train_matrix,
                  nrounds=nround,
                  nfold = cv.nfold,
                  verbose = FALSE,
                  prediction = TRUE)

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

names <-  colnames(x_train)
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
#head(importance_matrix)
write.csv(importance_matrix,file=paste0('~/Dropbox (XinghuaLu)/manu_material_final/sga_deg/sgaTDIset_predict_degTri_noCut_xgb_coef/coef_',deg,'.csv'),row.names = FALSE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = y_train)
head(OOF_prediction)

cm_cv <- confusionMatrix(factor(OOF_prediction$label,levels = c('0','1','2')), 
                         factor(OOF_prediction$max_prob-1,levels = c('0','1','2')),
                         mode = "everything")



# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = num_classes,
                          ncol=length(test_pred)/num_classes) %>%
  t() %>%
  data.frame() %>%
  mutate(label = y_test,
         max_prob = max.col(., "last"))
#confusion matrix of test set

cm_test = confusionMatrix(factor(test_prediction$label,levels = c('0','1','2')),
                          factor(test_prediction$max_prob-1,levels = c('0','1','2')),
                          mode = "everything")

# or predict separately

library(glmnet)
library(dplyr)
# example: https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
library(stringr)
library(ROCR)
library(caret)
library(ggplot2)

bn_df = read.csv('~/Documents/pbmc_benchmark_ae_bn_sparse_6000_cellmarkers.csv',stringsAsFactors = FALSE,row.names = 1)
cv_rmse = c()
test_rmse = c()
cv_r2 = c()
test_r2 = c()

for(i in colnames(ct_df_norm)){
  
  x = data.matrix(bn_df)
  y = data.matrix(ct_df_norm[,i])
  
  set.seed(1234) # fixed results
  splitIndex <- createDataPartition(y,p=0.75,list=FALSE,times=1)
  x_train <- x[splitIndex,]
  x_test <- x[-splitIndex,]
  y_train <- y[splitIndex]
  y_test <- y[-splitIndex]
  
  cvfit = cv.glmnet(x_train,y_train,family = "gaussian",type.measure ="mse",intercept=FALSE)
  
  #coefs_df = as.matrix(coef(cvfit,s='lambda.min'))
  #coefs = names(coefs_df[order(abs(coefs_df[,1]),decreasing = TRUE),])[1:40]
  #coefs = coefs[coefs!="(Intercept)"]
  train_pred = predict(cvfit,newx=x_train,s='lambda.min',type = "response")
  y_test = as.matrix(y_test)
  test_pred = predict(cvfit,newx=x_test,s='lambda.min',type = "response")
  
  cv_rmse = c(cv_rmse,sqrt(sum((train_pred-y_train)^2)/length(y_train))) 
  cv_r2 = c(cv_r2,1-sum((y_train-train_pred)^2)/sum((y_train-mean(y_train))^2))
  
  test_rmse = c(test_rmse,sqrt(sum((test_pred-y_test)^2)/length(y_test)))    
  test_r2 = c(test_r2,1-sum((y_test-test_pred)^2)/sum((y_test-mean(y_test))^2))
  
}






