# spatial transcriptome analysis

## Problem setting: 

single cell spatial transcriptomic data, each spot usually covers >1 cells, to infer potential cellular interactions, we need to infer the cell components at the spots as well as the gene sets which indicate functional modules describing cellular states. 

## Data preparation: 

- Simulation: with scRNAseq of known celltypes
- Visium: Brain, Breast cancer, 
- DBiT-seq: 
  - 10t(mouse embryo, day 10, 10 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4096261
  - 0725e10cL(mouse embryo, day 10, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4096262
  - 50t(mouse embryo, day 10, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4189611
  - 0628cL(mouse embryo, day 12, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4189612
  - 0702cL(mouse embryo, day 10, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4189613
  - 0713cL(mouse embryo, day 10, 25 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4189614
  - 0719cL(mouse embryo, day 10, 10 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4189615
  - 0702aL(mouse embryo, day 10, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4202307
  - 0713aL(mouse embryo, day 10, 25 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4202308
  - 0719aL(mouse embryo, day 10, 10 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4202309
  - 0725e10aL(mouse embryo, day 10, 50 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4202310
  - E11-1L(mouse embryo, day 11, 25 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4364242
  - E11-2L(mouse embryo, day 11, 25 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4364243
  - E11-FL-1L(mouse embryo, day 11, 10 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4364244
  - E11-FL-2L(mouse embryo, day 11, 10 micron): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4364245


  - FFPE-1(Mouse embryo E10.5, 25 um): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4745615
  - FFPE-2(Mouse embryo E10.5, 25 um): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4745616
  - Aorta(C57BL/6, 25um): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4745617
  - Atrium(C57BL/6, 25um): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4745618
  - Ventricle(C57BL/6, 25um): https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4745619


  - midbrain CTL(10 micron)
  - midbrain Treated(10 micron)


## Models:

- Autoencoder
Use spatial transcriptomic data as input (matrix dimension: spot by gene), use the bottle neck neurons as reconstructed features, we hope these could capture gene sets represent specific functional modules.
- Variational autoencoder 
- infinite mixture of VAEs
- semi-supervised learning
- image super-resolution

## Installation
```bash
# install
python setup.py install --user

# uninstall
pip uninstall bcvae

# re-install
./reinstall.sh
```
