
import numpy as np
import pandas as pd
import gene
from sergio import *
import matplotlib.pyplot as plt
import anndata

def read_file(path):
    with open(path, "rt") as f:
        return f.read()

def write_file(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

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

def input_file_regs_format(master_regulators, target_ids):
    input_file_regs = str()
    for interaction_type in master_regulators:
        for reg in master_regulators[interaction_type]:
            input_file_regs += str(target_ids[reg]) + ","
            for basal_int in master_regulators[interaction_type][reg][1:]:
                input_file_regs += str(basal_int) + ","
            input_file_regs = input_file_regs[:-1]
            input_file_regs += "\n"
    write_file("input_file_regs", input_file_regs[:-1])

# Identify master regulators from gene regulatory network
def find_master_regulators(interaction_pairs, number_bins):
    master_regulators = dict()
    master_regulators["receptor-TF"] = dict()
    master_regulators["no-ligand"] = dict()
    num_master_regs = 0

    for line in interaction_pairs.splitlines():
        items = line.split(",")
        src = items[0]
        dest = items[1]
        interaction_type = items[2]

        if interaction_type == "receptor-TF" or interaction_type == "no-ligand":
            if interaction_type == "receptor-TF":
                corr = src + "_"
            else: corr = src[:-1]
            if src not in master_regulators[interaction_type]:
                num_master_regs += 1
            master_regulators[interaction_type][src] = [corr]
            if dest in master_regulators[interaction_type]:
                master_regulators.pop(dest)
                num_master_regs -= 1
    
    # Generate random basal production rates for each cell
    for reg in master_regulators["receptor-TF"]:
        dummy_reg = master_regulators["receptor-TF"][reg][0]
        for i in range(number_bins):
            if (i < number_bins//2):
                basal_prod = np.round(np.random.uniform(0, 1), 2)
                master_regulators["receptor-TF"][reg].append(basal_prod)
                master_regulators["no-ligand"][dummy_reg].append(0)
            else:
                index = 1 + (i - number_bins//2) # corresponding cell for reg
                basal_prod = np.round((1 - master_regulators["receptor-TF"][reg][index]), 2)
                master_regulators["receptor-TF"][reg].append(0)
                master_regulators["no-ligand"][dummy_reg].append(basal_prod)

    return master_regulators, num_master_regs

# Choose enumerations for regulators and targets
def choose_target_ids(interaction_pairs, master_regulators, num_master_regs):
    target_id = num_master_regs
    master_id = 0
    target_ids = dict()
    for line in interaction_pairs.splitlines():
        items = line.split(",")
        src = items[0]
        dest = items[1] 
        interaction_type = items[2]

        if (src not in target_ids and (src in master_regulators["receptor-TF"] 
            or src in master_regulators["no-ligand"])):
            target_ids[src] = master_id
            master_id += 1
        if src not in target_ids:
            target_ids[src] = target_id
            target_id += 1
        if dest not in target_ids:
            target_ids[dest] = target_id
            target_id += 1
    return target_ids

# Build dictionary with information about targets
def build_input_file_targets(interaction_pairs, target_ids, hill_coeff, interaction_strength):
    targets = dict()
    genes = set()
    for line in interaction_pairs.splitlines():
        items = line.split(",")
        src = items[0]
        dest = items[1]
        interaction_type = items[2]
        if (interaction_type == "TF-target_gene"): 
            continue

        genes.add(dest)
        if (interaction_type == "no_ligand"): 
            interaction_strength = 0
        else: 
            interaction_strength = np.round(np.random.uniform(0, 1), 2)
        
        if dest in targets:
            targets[dest]["reg_count"] += 1.0
            targets[dest]["reg_ids"].append(target_ids[src])
            targets[dest]["interaction_strengths"].append(interaction_strength)
            targets[dest]["hill_coeffs"].append(hill_coeff)
        else:
            targets[dest] = dict()
            targets[dest]["target_id"] = target_ids[dest]
            targets[dest]["reg_count"] = 1.0
            targets[dest]["reg_ids"] = [target_ids[src]]
            targets[dest]["interaction_strengths"] = [interaction_strength]
            targets[dest]["hill_coeffs"] = [hill_coeff]
    return targets, len(genes)

def input_file_format(path, number_bins, hill_coeff, interaction_strength):
    interaction_pairs = read_file(path)

    # Identify master regulators from gene regulatory network
    master_regulators, num_master_regs = find_master_regulators(interaction_pairs, number_bins)
    # Choose enumerations for regulators and targets
    target_ids = choose_target_ids(interaction_pairs, master_regulators, num_master_regs)
    # Build dictionary with information about targets
    targets, num_genes = build_input_file_targets(interaction_pairs, target_ids, hill_coeff, interaction_strength)
    # Format and write the input files
    input_file_targets = input_file_targets_format(targets)
    input_file_regs = input_file_regs_format(master_regulators, target_ids)

    total_genes = num_master_regs + num_genes
    return total_genes

##############################################
# SERGIO Functions
##############################################


def steady_state_clean_data(number_genes, number_bins):
    sim = sergio(number_genes=number_genes, number_bins = number_bins, number_sc = 300, 
                 noise_params = 1, decays=0.8, sampling_state=15, 
                 noise_type='dpd')
    print('\nBuilding graph...\n')
    sim.build_graph(input_file_taregts ='input_file_targets', 
                    input_file_regs='input_file_regs', 
                    shared_coop_state=2)
    print('\nSimulating...\n')
    sim.simulate()
    print('\nSimulation complete! Adding technical noise...\n')
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis = 1)
    return sim, expr, expr_clean

def steady_state_technical_noise(sim, expr):
    # Add outlier genes
    expr_O = sim.outlier_effect(expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    # Add Library Size Effect
    libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean = 4.6, scale = 0.4)
    # Add Dropouts
    binary_ind = sim.dropout_indicator(expr_O_L, shape = 6.5, percentile = 82)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    # Convert to UMI count
    count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)
    # Make a 2d gene expression matrix
    count_matrix = np.concatenate(count_matrix, axis = 1)
    return count_matrix

##############################################
# h5ad Read and Write
##############################################

def open_h5ad(filename):
    # Replace 'your_file.h5ad' with the path to your h5ad file
    anndata_obj = anndata.read_h5ad(filename)
    print(anndata_obj)
    print(anndata_obj.X)

def make_h5ad(expression_data):
    cell_ids = [f"Cell {i}" for i in range(len(expression_data))]
    gene_ids = [f"Gene {j}" for j in range(len(expression_data[0]))]
    anndata_obj = anndata.AnnData(X=expression_data, 
                                  obs={'Cell ID': cell_ids}, 
                                  var={'Gene': gene_ids})
    anndata_obj.write_h5ad("gene_expression.h5ad")

##############################################
# main
##############################################

def run(interaction_pairs, number_bins=2, diff=False, bMat=None, 
        hill_coeff=1.0, interaction_strength=1.0):
              
    number_genes = input_file_format(interaction_pairs, number_bins, 
                                     hill_coeff, interaction_strength)

    if (bMat == None): 
        if (number_bins == 1): bMat = 'bMat/1_cell_type.tab'
        elif (number_bins == 2): bMat = 'bMat/2_cell_types.tab'
        elif (number_bins == 3): bMat = 'bMat/3_cell_types.tab'
        elif (number_bins == 4): bMat = 'bMat/4_cell_types.tab'

    if (diff): 
        differentiation_clean_data(number_genes, number_bins, bMat)
    else: 
        sim, expr, expr_clean = steady_state_clean_data(number_genes, number_bins)
        count_matrix = steady_state_technical_noise(sim, expr)
        make_h5ad(count_matrix)
        open_h5ad("gene_expression.h5ad")