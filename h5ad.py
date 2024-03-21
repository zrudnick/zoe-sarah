
import anndata as ad

##############################################
# h5ad Read and Write
##############################################

def open_h5ad(path):
    adata = ad.read_h5ad(path)
    return adata

def gene_expr_to_h5ad(gene_expr, ad_path, n_sc):
    gene_ids = [i for i in range(len(gene_expr))]
    cell_ids = [j for j in range(len(gene_expr[0]))]
    cell_types = [(cell_id // n_sc) for cell_id in cell_ids]
    # print(cell_ids)
    # print(cell_types)
    # gene_names 
    adata = ad.AnnData(X=gene_expr, 
                        obs={'Gene': gene_ids}, 
                        var={'Cell Type': cell_types})
    #, 'Gene Name': gene_names
    adata.write_h5ad(ad_path)
    return adata

def gene_expr_to_h5ad_with_gene_names(gene_expr, ad_path, n_sc, target_ids):
    gene_ids = [i for i in range(len(gene_expr))]
    # print(gene_ids)
    # print(target_ids)
    cell_ids = [j for j in range(len(gene_expr[0]))]
    cell_types = [(cell_id // n_sc) for cell_id in cell_ids]
    gene_names=[]
    for g in cell_ids: 
        for k, v in target_ids.items():
            if g == v: 
                gene_names.append(k)
    adata = ad.AnnData(X=gene_expr, 
                        obs={'Gene': gene_ids, 'Gene Names': gene_names}, 
                        var={'Cell Type': cell_types})
    # adata.obs.index = gene_names
    # print("ADATA 1 OBS INDEX", adata.obs.index)
    # print("ADATA GENE NAMES", adata.obs['Gene Names'])
    # print(adata.obs.index)
    #, 'Gene Name': gene_names
    adata.write_h5ad(ad_path)
    return adata