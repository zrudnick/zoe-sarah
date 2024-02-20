
import scanpy as sc
import anndata as ad

from format_input_files import *
from sergio_steady_state import *
from h5ad import *
from spatial import *

##############################################
# main
##############################################

def run_spatial(path, n_genes, n_bins, n_sc, T=0.5, lr=5):
    # T is the temperature to use
    # lr is the ratio of cells with receptors to ligands
 
    print("--------------------------------------")
    print("   Determining spatial coordinates    ")
    print("--------------------------------------")

    simulate_spatial_expression_data(path, n_genes, n_bins, n_sc, T, lr)

def run_umap(path, n_neighbors=50, min_dist=0.01):
    print("--------------------------------------")
    print("         Creating UMAP graph          ")
    print("--------------------------------------")

    adata = open_h5ad(path)
    adata = ad.AnnData(X=adata.X.T, obs=adata.var, var=adata.obs)
    print(f"Gene Expression Matrix: {adata.X.shape[0]} Single Cells, {adata.X.shape[1]} Genes")

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata, min_dist=min_dist)
    sc.pl.umap(adata, color=["Cell Type", "16", "17", "18", "19"])

def run_sergio(interaction_pairs, n_sc=300, hill_coeff=1.0, interaction_strength=1.0, basal_prod_type=""):
    n_genes, n_master_regs, n_bins = input_file_format(interaction_pairs, hill_coeff, interaction_strength, basal_prod_type)
    ad_path = "gene_expression.h5ad"

    print("--------------------------------------")
    print("            Running SERGIO            ")
    print("--------------------------------------")
    sim, expr = steady_state_clean_data(n_genes, n_bins, n_sc)
    gene_expr = steady_state_technical_noise(sim, expr, n_genes, n_master_regs)
    #add_dummy_counts(gene_expr, n_master_regs)
    adata = gene_expr_to_h5ad(gene_expr, ad_path, n_sc)

    return n_genes, n_bins, n_sc

n_genes, n_bins, n_sc = run_sergio("interaction_pairs/interaction_pairs_v3_small.csv")
# run_umap("gene_expression.h5ad")
run_spatial("gene_expression.h5ad", n_genes, n_bins, n_sc)

# add ligand to gene expression matrix
# add 1 for ligand gene and 0 for other genes
# get correspondance list of master regulator to cell type
# or rename things