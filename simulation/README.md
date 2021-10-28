### Running simulation

```
# Simulate synthetic ST data from PBMC scRNA-seq
./st_simulate.R --n_spots 2000 -o data/

# Calculate gsva score from synthetic ST data & signature file
gsva_fname="markers_gsva_pbmc.csv"
./gsva.R --exp data/pbmc_benchmark_spot_exp.csv --sig data/pbmc_markers_gsva_mod.csv -o data/ --name $gsva_fname
```
