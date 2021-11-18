#!/usr/bin/env Rscript
# Simulate synthetic Spatial Transcriptomics matrix from PBMC scRNA-seq data

rm(list=ls())
   
suppressMessages(library(data.table))
suppressMessages(library(MCMCpack))
suppressMessages(library(optparse))
suppressMessages(library(scater))
suppressMessages(library(testit))
suppressMessages(library(optparse))


#' Simulation synthetic spatial transcriptomics matrix from PBMC scRNA-seq dataset
#'
#' @param meta metadata from PBMC scRNA-seq data
#' @param exp expression count from PBMC scRNA-seq data
#' @param n_spots number of spots in the matrix
#' @param alpha dropout rate 
#' @param nmin min. number of cells per spot 
#' @param nmax max. number of cells per spot
#' 
#' return list of synthetic expression [S x G] & ground-truth cell-type composition for each spot [S x C]
simSynthST <- function(meta, exp, n_spots, alpha = 0.5, nmin = 5, nmax = 10) {
  cts <- unique(meta$nnet2)
  n_cts <- length(cts)
  n_cells <- seq(nmin, nmax)
  ct_exps <- lapply(as.list(unique(meta$nnet2)), function(x) exp[, meta$nnet2 == x])  # expression subsetted by cell types
  
  syn_exp <- matrix(0, nrow = n_spots, ncol = nrow(exp))
  syn_exp <- data.frame(syn_exp, row.names =  paste0("s", seq(1, n_spots)))
  colnames(syn_exp) <- row.names(exp)
  
  syn_ct <- matrix(0, nrow = n_spots, ncol = length(cts))
  syn_ct <- data.frame(syn_ct, row.names =  paste0("s", seq(1, n_spots)))
  colnames(syn_ct) <- cts
  
  cat("Simulating synthetic ST matrix...\n")
  pb <- txtProgressBar(min = 0, max = n_spots, initial = 0, style = 3)
  
  for (s in 1:n_spots) {
    setTxtProgressBar(pb, s)
    
    n_cell <- sample(n_cells, 1) # sample # cell mixture for each spot
    cell_count <- round(n_cell * rdirichlet(1, rep(1, n_cts)))
    exp_samp <- do.call(
      cbind, 
      mapply(function(mat, n) mat[, sample(ncol(mat), size = n), drop = FALSE], ct_exps, cell_count)
    )
    dropout <- rbinom(nrow(exp_samp), 1, alpha)
    syn_exp[s, ] <- dropout * as.numeric(rowSums(exp_samp))
    
    spot_ct <- table(meta[colnames(exp_samp), "nnet2"])
    syn_ct[s, names(syn_ct) %in% names(spot_ct)] = spot_ct
  }
  close(pb)
  
  return (list("exp" = syn_exp, "ct" = syn_ct))
}


###############################

option_list <- list(
  make_option(c("--n_spots"), type = "integer", default = 1000, 
              help = "Number of synthetic spots"),
  make_option(c("-a", "--alpha"), type = "double", default = 0.5,
              help = "Dropout rate"),
  make_option(c("-o", "--out"), type = "character", default = "data/",
              help = "Output directory", metavar = "character")
)

opt_parser = OptionParser(option_list = option_list);
opt = parse_args(opt_parser)

# Load PBMC scRNA-seq datasets
n_spots <- opt$n_spots
alpha <- opt$alpha
data_path <- opt$out

ifile <- "sce.all_classified.technologies.RData"
assert("Input file doesn't exist", file.exists(paste0(data_path, ifile)))
assert("Dropout rate must be within (0, 1)", (alpha >= 0 && alpha <= 1))
cat("Loading PBMC datasets...\n")
load(file=paste0(data_path, ifile))

# Select dataset from Smart-seq2 protocol
sce_smart <- sce[, sce$batch == "Smart-Seq2"]
sce_smart$nnet2 <- as.character(sce_smart$nnet2)
meta <- data.frame(sce_smart@colData)
exp <- sce_smart@assays$data$counts
cat("Cell types from PBMC smart-seq:\n")
unique(meta$nnet2)

syn <- simSynthST(meta, exp, n_spots)

# Save to data directory
cat(paste0("Saving synthetic data to ", normalizePath(data_path), " ...\n"))
write.csv(syn$exp, file=paste0(data_path, "pbmc_benchmark_spot_exp.csv"))
write.csv(syn$ct, file=paste0(data_path, "pbmc_benchmark_spot_ct.csv"))
