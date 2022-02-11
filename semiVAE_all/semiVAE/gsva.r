library(GSVA)
library(Seurat)



gsva_fun<- function(sample_id, file_exp, file_gene_sig){

print(paste0('run gsva for ',sample_id))
X_exp <- read.table(file_exp, header = TRUE, sep= ",", row.names = 1)
X_exp <- t(data.matrix(X_exp))
p <- dim(X_exp)[1] # number of genes
n <- dim(X_exp)[2] # number of samples
gene_sig <- read.table(file_gene_sig, header = TRUE, sep= ",",)
gs_celltype <- as.list(gene_sig)
gs_celltype <- lapply(gs_celltype, function(x) x[nzchar(x)])
gsva.es <- gsva(X_exp, gs_celltype, verbose=FALSE)
return(gsva.es)                    
}