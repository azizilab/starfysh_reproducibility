#!/usr/bin/env Rscript
# Calculate ST GSVA score from expression & signature files

rm(list=ls())

suppressMessages(library(GSVA))
suppressMessages(library(testit))
suppressMessages(library(optparse))

option_list <- list(
  make_option(c("--exp"), type = "character", 
              help = "Input Spot x Gene expression file"),
  make_option(c("--sig"), type = "character",
              help = "Input signature file"),
  make_option(c("-o", "--out"), type = "character", default = "data/",
              help = "Output directory", metavar = "character"),
  make_option(c("--name"), type = "character", default = "gsva.csv",
              help = "Output file name")
)

opt_parser = OptionParser(option_list = option_list);
opt = parse_args(opt_parser)

assert("Expresion / Signature input file names is missing", !is.null(opt$exp) && !is.null(opt$sig))
assert("Expression input file doesn't exist", file.exists(opt$exp))
assert("Signature input file doesn't exist", file.exists(opt$sig))

if (!dir.exists(opt$out)) {
  mkdir(opt$out)
}

cat("Loading expression & signature files...\n")
exp_df <- read.csv(opt$exp, row.names = 1)
# exp_df = read.csv('~/Downloads/bc_visium_mod/exp_markers_df_gsva_pbmc_x.csv',row.names = 1)
# exp_df[1:4,1:4]

sigfile <- read.csv(opt$sig, header = T)
# sigfile = read.csv('~/Downloads/bc_visium_mod/pbmc_markers_gsva_mod.csv',header=T)
sigfile$Cell.type = as.character(sigfile$Cell.type)
sigfile$Symbol = as.character(sigfile$Symbol)

#cat("Cell types in signature file:\n")
table(sigfile$Cell.type)

genelist = split(as.character(sigfile[,2]),as.character(sigfile[,1]))

gsva_scores = gsva(t(exp_df),genelist,method="ssgsea")
gsva_scores = gsva(t(exp_df),genelist,method="gsva")
# heatmap(gsva_scores)
# heatmap(gsva_scores)

write.csv(t(gsva_scores), file = paste0(opt$out, opt$name))
# write.csv(t(gsva_scores), file = '~/Downloads/bc_visium_mod/markers_gsva_pbmc_x.csv')
