
import numpy as np
import pandas as pd
import gene
from sergio import *
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc

# Read a file from the input path
def read_file(path):
    with open(path, "rt") as f:
        return f.read()

# Write a file to the input path
def write_file(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

# Create dict with the form {interaction_type: (src_gene, dest_gene)}
def to_dict(interaction_pairs):
    d = dict()
    for line in interaction_pairs.splitlines():
        items = line.split(",")
        src = items[0]
        dest = items[1]
        interaction_type = items[2]
        if interaction_type not in d:
            d[interaction_type] = []
        d[interaction_type].append((src, dest))
    return d

# Use targets dictionary to write input_file_targets
def input_file_targets_format(targets):
    input_file_targets = str()
    for dest in targets:
        input_file_targets += str(targets[dest]["target_id"]) + ","
        input_file_targets += str(targets[dest]["reg_count"]) + ","
        for reg_id in targets[dest]["reg_ids"]:
            input_file_targets += str(reg_id) + ","
        for interaction_strength in targets[dest]["interaction_strengths"]:
            input_file_targets += str(interaction_strength) + ","
        for hill_coeff in targets[dest]["hill_coeffs"]:
            input_file_targets += str(hill_coeff) + ","
        input_file_targets = input_file_targets[:-1]
        input_file_targets += "\n"
    write_file("input_file_targets", input_file_targets[:-1])

# Use master_regs dictionary to write input_file_regs
def input_file_regs_format(master_regs, target_ids):
    input_file_regs = str()
    for interaction_type in master_regs:
        for reg in master_regs[interaction_type]:
            input_file_regs += str(target_ids[reg]) + ","
            for basal_int in master_regs[interaction_type][reg][1:]:
                input_file_regs += str(basal_int) + ","
            input_file_regs = input_file_regs[:-1]
            input_file_regs += "\n"
    write_file("input_file_regs", input_file_regs[:-1])

# Identify master regulators from the gene regulatory network
def find_master_regs(interaction_pairs):
    master_regs = dict()
    master_regs["receptor-TF"] = dict()
    master_regs["no-ligand"] = dict()
    n_master_regs = 0

    # Loop through every master regulator-TF pair
    for (src, dest) in interaction_pairs["receptor-TF"]:

       # Add potential master regulator
        if src not in master_regs["receptor-TF"]:
            corr = src + "_" # name of corresponding dummy reg
            master_regs["receptor-TF"][src] = [corr]
            master_regs["no-ligand"][corr] = [src]
            n_master_regs += 2

        # Not a master regulator -> remove from dict
        if dest in master_regs["receptor-TF"]:
            corr = dest + "_" # name of corresponding dummy reg
            master_regs["receptor-TF"].pop(dest)
            master_regs["no-ligand"].pop(corr)
            n_master_regs -= 2
    return master_regs, n_master_regs

# Add dummy nodes to interaction_pairs
def add_dummy_nodes(interaction_pairs, master_regs):
    interaction_pairs["no-ligand"] = [] # create key for dummy regs
    n_null = 0
    for dummy_src in master_regs["no-ligand"]:
        # Add dummy node to interaction_pairs
        null_dest = "null"+str(n_null)

        # The destination of null simulates when there is no downstream activity
        if (dummy_src, null_dest) not in interaction_pairs["no-ligand"]:
            interaction_pairs["no-ligand"].append((dummy_src, null_dest)) # point to null
        n_null += 1
    return interaction_pairs

# Choose enumerations for regs and targets
def choose_target_ids(interaction_pairs, master_regs, n_master_regs):
    target_id = n_master_regs
    master_id = 0
    target_ids = dict()
    for interaction_type in interaction_pairs:
        # Error if this loops through TF-target_gene but doesn't add to dict
        if interaction_type == "TF-target_gene": continue

        for (src, dest) in interaction_pairs[interaction_type]:
            is_master_regulator = (src in master_regs["receptor-TF"] or src in master_regs["no-ligand"])
            if (src not in target_ids and is_master_regulator):
                target_ids[src] = master_id
                master_id += 1
            
            if src not in target_ids:
                target_ids[src] = target_id
                target_id += 1
            if dest not in target_ids:
                target_ids[dest] = target_id
                target_id += 1
    return target_ids

# Generate basal production rate for each cell (default 1.0 for every cell)
def generate_basal_prod_rates(n_bins, basal_prod_type):
    basal_prod_rates = []
    for i in range(n_bins):
        if basal_prod_type == "Uniform Distribution":
            basal_prod_rates.append(np.round(np.random.uniform(0, 1), 2))
        else:
            basal_prod_rates.append(1.0)
    return basal_prod_rates

# Add basal production rates to master_regs dictionary
def write_basal_prod_rates(master_regs, n_bins, target_ids, basal_prod_rates):
    zero_rate = 0.0001 # non-zero value to support SERGIO algorithms
    n_real_bins = n_bins//2

    # Loop through all master regulators
    for reg in master_regs["receptor-TF"]:
        dummy_reg = master_regs["receptor-TF"][reg][0] # get corresponding dummy regulator

        # Generate random basal production rates for each cell
        for i in range(n_bins):
            if (i < n_real_bins): # cells for master regulators

                # Each master regulator is expressed only in cell with corresponding ID
                if (i == target_ids[reg]): 
                    basal_prod = basal_prod_rates[i]
                else: basal_prod = 0.0001
                master_regs["receptor-TF"][reg].append(basal_prod)
                master_regs["no-ligand"][dummy_reg].append(zero_rate)
            else: # cells for dummy nodes

                index = 1 + (i - n_bins//2) # index of cell expressed by corresponding regulator
                basal_prod = np.round((master_regs["receptor-TF"][reg][index]), 4)
                master_regs["receptor-TF"][reg].append(zero_rate)
                master_regs["no-ligand"][dummy_reg].append(basal_prod)

# Build dictionary with target information for each regulator
def build_input_file_targets(interaction_pairs, target_ids, hill_coeff, interaction_strength):
    targets = dict()
    genes = set()
    for interaction_type in interaction_pairs:
        for (src, dest) in interaction_pairs[interaction_type]:
            if (interaction_type == "TF-target_gene"): continue

            genes.add(dest)
            
            if interaction_type == 'no_ligand': interaction_strength = 5.0
            if dest in targets:
                targets[dest]["reg_count"] += 1
                targets[dest]["reg_ids"].append(target_ids[src])
                targets[dest]["interaction_strengths"].append(interaction_strength)
                targets[dest]["hill_coeffs"].append(hill_coeff)
            else:
                targets[dest] = dict()
                targets[dest]["target_id"] = target_ids[dest]
                targets[dest]["reg_count"] = 1
                targets[dest]["reg_ids"] = [target_ids[src]]
                targets[dest]["interaction_strengths"] = [interaction_strength]
                targets[dest]["hill_coeffs"] = [hill_coeff]
    return targets, genes

# Create input_file_targets and input_file_regs files
def input_file_format(path, hill_coeff, interaction_strength, basal_prod_type):
    interaction_pairs = read_file(path)

    # Convert csv to dictionary in form {interaction_type: (src_gene, dest_gene)}
    interaction_pairs = to_dict(interaction_pairs)

    # Identify master regs from gene regulatory network
    master_regs, n_master_regs = find_master_regs(interaction_pairs)

    # Add dummy nodes to interaction_pairs dictionary
    interaction_pairs = add_dummy_nodes(interaction_pairs, master_regs)
    n_bins = n_master_regs

    # Choose enerations for regs and targets
    target_ids = choose_target_ids(interaction_pairs, master_regs, n_master_regs)

    # Generate basal production rates in each cell
    basal_prod_rates = generate_basal_prod_rates(n_bins, basal_prod_type)
    write_basal_prod_rates(master_regs, n_bins, target_ids, basal_prod_rates)

    # Build dictionary with information about targets
    targets, genes = build_input_file_targets(interaction_pairs, target_ids, hill_coeff, interaction_strength)

    # Format and write the input files
    input_file_targets = input_file_targets_format(targets)
    input_file_regs = input_file_regs_format(master_regs, target_ids)

    # Calculate total number of genes
    n_genes = n_master_regs + len(genes)

    return n_genes, n_master_regs, n_bins

##############################################
# SERGIO Functions
##############################################


def steady_state_clean_data(n_genes, n_bins, n_sc):
    sim = sergio(number_genes=n_genes, number_bins = n_bins, number_sc = n_sc, 
                 noise_params = 1, decays=0.8, sampling_state=15, 
                 noise_type='dpd')
    print('Building graph...')
    sim.build_graph(input_file_taregts ='input_file_targets', 
                    input_file_regs='input_file_regs', 
                    shared_coop_state=2)
    print('Simulating...')
    sim.simulate()
    print('Simulation complete! Adding technical noise...')
    expr = sim.getExpressions()
    
    return sim, expr

def steady_state_technical_noise(sim, expr, n_genes, n_master_regs):
    # Add outlier genes
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    # Add Library Size Effect
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
    # Add Dropouts
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 6.5, percentile = 82)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    # Convert to UMI count
    gene_expr = sim.convert_to_UMIcounts(expr_O_L_D)
    # Make a 2d gene expression matrix
    gene_expr = np.concatenate(gene_expr, axis = 1)

    return gene_expr

##############################################
# h5ad Read and Write
##############################################

def open_h5ad(path):
    adata = ad.read_h5ad(path)
    return adata

def make_h5ad(gene_expr, ad_path, n_sc):
    gene_ids = [i for i in range(len(gene_expr))]
    cell_ids = [j for j in range(len(gene_expr[0]))]
    cell_types = [(cell_id // n_sc) for cell_id in cell_ids]
    adata = ad.AnnData(X=gene_expr, 
                        obs={'Gene': gene_ids}, 
                        var={'Cell Type': cell_types})
    adata.write_h5ad(ad_path)
    return adata
    
def add_dummy_counts(gene_expr, n_master_regs):
    dummy_i = n_master_regs//2
    for i in range(n_master_regs//2):
        gene_expr[i] += gene_expr[dummy_i]
        gene_expr = np.delete(gene_expr, dummy_i, axis=0)

##############################################
# main
##############################################

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
    sc.pl.umap(adata, color=["Cell Type", "0", "1", "2", "3", "4", "5", "6", "7"])

def run(interaction_pairs, n_sc=300, hill_coeff=1.0, interaction_strength=1.0, basal_prod_type=""):
    n_genes, n_master_regs, n_bins = input_file_format(interaction_pairs, hill_coeff, interaction_strength, basal_prod_type)
    ad_path = "gene_expression.h5ad"

    print("--------------------------------------")
    print("            Running SERGIO            ")
    print("--------------------------------------")
    sim, expr = steady_state_clean_data(n_genes, n_bins, n_sc)
    gene_expr = steady_state_technical_noise(sim, expr, n_genes, n_master_regs)
    #add_dummy_counts(gene_expr, n_master_regs)
    adata = make_h5ad(gene_expr, ad_path, n_sc)
