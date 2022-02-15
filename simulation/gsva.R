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

# Load expression & normalize by median libsize
cat("Loading expression & signature files...\n")
exp_df_raw <- read.csv(opt$exp, row.names = 1)
libsize <- colSums(exp_df_raw)
median.libsize <- median(libsize)
exp_df <- median.libsize * sweep(exp_df_raw, 2, libsize, FUN = '/')

gene_sig <- read.csv(opt$sig, header = T)
gs_celltype <- as.list(gene_sig)
gs_celltype <- lapply(gs_celltype, function(x) x[nzchar(x)])

gsva_scores <- gsva(t(exp_df), gs_celltype, method  = "gsva")

# Write to output
write.csv(t(gsva_scores), file = paste0(opt$out, opt$name))
