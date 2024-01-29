
from spatial import *

#n_genes, n_bins, n_sc = run_sergio("interaction_pairs/interaction_pairs_v3_small.csv")
# run_umap("gene_expression.h5ad")
run_spatial("gene_expression.h5ad", n_genes=50, n_bins=8, n_sc=300)