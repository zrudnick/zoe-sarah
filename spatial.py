
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
import os
from collections import Counter, defaultdict
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm, trange

import gene
from sergio import *

##############################################
# SERGIO Inputs from GRN
##############################################

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
        #if interaction_type == "TF-target_gene": continue

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
            #if (interaction_type == "TF-target_gene"): continue

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
# Spatial modeling functions
##############################################

# Load gene expression data from SERGIO  
def load_sc_data(path):
    # Load SC data
    sc_data = open_h5ad(path)
    sc_data = ad.AnnData(X=sc_data.X.T, obs=sc_data.var, var=sc_data.obs)
    return sc_data
        
# Determine random coordinates for each spot
def get_spatial_coordinates(n_spots):
    sqr_n_spots = int(n_spots**0.5)
    spatial_coordinates = []
    i_j = set()
    for i in range(sqr_n_spots):
        for j in range(sqr_n_spots):
            i_j.add((i, j))
    n_spots_new = 0
    while len(i_j) > 0:
        n_spots_new += 1
        (i_curr, j_curr) = list(i_j)[np.random.choice(len(i_j))]
        i_j.remove((i_curr, j_curr))
        spatial_coordinates.append((i_curr, j_curr))
    (i_curr, j_curr) = sqr_n_spots, 0
    while n_spots_new < n_spots:
        spatial_coordinates.append((i_curr, j_curr))
        (i_curr, j_curr) = (i_curr, j_curr + 1)
        n_spots_new += 1
        if j_curr == sqr_n_spots:
            (i_curr, j_curr) = (i_curr + 1, 0)
    return spatial_coordinates

# Generate an empty square spot graph to emulate Visium data
def generate_empty_spatial_image(n_bins, n_sc, sc_data):
    # Load Visium spatial image
    # st_data = sc.read_visium('c:/Users/zoeru/Downloads/STB01S1_preproccesed/STB01S1_preproccesed/outs')
    # st_data.var_names_make_unique()
    #print(st_data.uns['spatial'])

    # define dimensions of the empty Visium spatial image
    n_genes = 2000
    n_spots = n_bins * n_sc # should be n_sc * n_bins

    spatial_coordinates = get_spatial_coordinates(n_spots)

    # Create a blank AnnData object
    adata = sc.AnnData(
        X=np.zeros((n_spots, n_genes)),  # Placeholder for gene expression data
        obs=pd.DataFrame(index=np.arange(n_spots)),  # Observation metadata
        var=pd.DataFrame(index=np.arange(n_genes))  # Variable (gene) metadata
    )
    adata.obsm['spatial'] = np.asarray(spatial_coordinates) # Spatial coordinates

    # Additional metadata, if needed
    adata.obs['in_tissue'] = [0] * n_spots
    adata.obs['array_row'] = [0] * n_spots
    adata.obs['array_col'] = [0] * n_spots
    adata.var['gene_ids'] = [0] * n_genes
    adata.var['feature_types'] = [0] * n_genes
    adata.var['genome'] = [0] * n_genes

    # Print the AnnData object
    return adata, n_spots

# Compute distance matrix as Kernel   
# Describes how similar two spots should be based on distance  
def distance_matrix(st_data):

    # Compute K, the covariance matrix
    xy = st_data.obsm['spatial']
    dist = squareform(pdist(xy))                   # spot pair distances
    sigma = np.median(dist)                        # median value of spot pair distances
    lamb = 0.1                                     # similarity to smooth GP, set to 0.1 by default
    K = np.exp(- dist ** 2 / (lamb * sigma ** 2))
    return K, xy

# Plot the empty square spot graph
def plot_st_data(st_data):
    spatial_coordinates = st_data.obsm['spatial']

    # Visualize spatial distribution
    plt.figure(figsize=(8, 8))
    plt.scatter(spatial_coordinates[:, 0], spatial_coordinates[:, 1], s=50, alpha=0.5)
    plt.title('Spatial Distribution of Visium Spots')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    return

# Use K as the Kernel matrix of a GP to generate samples for each cell type
def get_gaussian_process_samples(K, n_bins, n_spots):
    mean = [0]*n_spots
    cov = K
    # gp stands for Gaussian Process
    gp_samples = np.random.multivariate_normal(mean, cov, size=(n_bins,)).T # gp sample for each spot
    return gp_samples

# I don't really get this yet
# Aka how or why to implement it for our sake
def set_temperature(st_data, gp_samples, n_bins, n_sc):
    # T is the temperature
    # A small value of T tends to preserve the dominant cell type with the highest energy
    # while a large value of T maintains the original cell type proportions
 
    cell_abundance = n_sc / (n_sc * n_bins) # code below expects dataframe?
   
    # try T 1-5 for now
    for T in [1, 5]:
        # the cell type proportion at every spot is an 'energy' Phi
        columns = range(n_bins)
        columns = [str(col) for col in columns]
        index = st_data.obs.index
        phi_cell = pd.DataFrame(gp_samples, columns=columns, index=index)
        # each phi_cell is a cell-type energy vector aligned with spots
        # the proportion Pi_c_s for each spot and each cell type is then calculated using the energy
        pi_cell = (cell_abundance * np.exp(phi_cell/T)).div((cell_abundance * np.exp(phi_cell/T)).sum(1),axis='index')
        st_data.obsm['pi_cell' + str(T)] = pi_cell # add to existing data

        for cell_type, pi in pi_cell.items():
            st_data.obs[str(cell_type) + '_' + str(T)] = pi_cell[cell_type]
    print(st_data.obsm['pi_cell1'])
    print(st_data.obsm['pi_cell5'])
    return st_data

# Format gene expression data by cell groups
def determine_cell_groups(st_data, sc_data):

    # Queues of each cell type
    celltype_order = st_data.obsm['pi_cell1'].columns.tolist() # list of cell types
    cell_groups = [x for x in sc_data.to_df().groupby(sc_data.obs['Cell Type'],sort=False)] 
    cell_groups.sort(key=lambda x: celltype_order.index(str(x[0])),)
    cell_groups = [(ct,cell_group.sample(frac=1)) for   ct, cell_group in cell_groups]
    cell_groups_remains = pd.Series([y.shape[0] for x,y in cell_groups],index = [x for x,y in cell_groups])
    return cell_groups

# Sample a cell type
def sample_celltype(cell_groups, cell_type_index, n):
    cell_type, type_df = cell_groups[cell_type_index]
    pop_df = type_df.iloc[:n]
    type_df.drop(pop_df.index, inplace=True)
    type_tags = pop_df.index.tolist()
    type_sum = pop_df.sum(0)
    
    if len(type_tags) == 0:
        print(f'Warning: {cell_type} has no more cells')
    
    return n, type_tags, [str(cell_type)] * n, type_sum

# Synthesize the data spot by spot
def synthesize_data(cell_groups, st_data, sc_data, xy, n_bins, n_spots):

    # Create AnnData objects for simulation information
    st_simu = sc.AnnData(np.zeros((st_data.shape[0],sc_data.shape[1])),obs=pd.DataFrame(index=st_data.obs.index,columns=['cell_counts','cell_tags','cell_types']))

    # Choose cell type for each spot
    for i in trange(n_spots):
        spot_size = 1
        prob_in_spot = st_data.obsm["pi_cell5"].iloc[i].values
        choice = np.random.choice(n_bins, spot_size, p=prob_in_spot)
        # number can be from 0 to n_bins-1
        
        spot_size_true = 0
        spot_tags = []
        spot_types = []
        spot_X = np.zeros(sc_data.shape[1])
        
        cell_type_index = choice[0]
        spot_size_ct, type_tags, type_list, type_sum = sample_celltype(cell_groups, cell_type_index, 1)
            
        spot_size_true += spot_size_ct
        spot_tags.extend(type_tags)
        spot_types.extend(type_list)
            
        spot_X += type_sum 
        
        st_simu.obs.iloc[i]['cell_counts'] = spot_size_true
        st_simu.obs.iloc[i]['cell_tags'] = ','.join(spot_tags)
        st_simu.obs.iloc[i]['cell_types'] = ','.join(spot_types)
        st_simu.X[i] = spot_X

    st_simu.obsm['spatial'] = st_data.obsm['spatial']
    st_simu.obs['cell_counts'] = st_simu.obs['cell_counts'].astype(str)
    st_simu.var_names = sc_data.var_names

    mapping = st_simu.obs['cell_tags'].str.split(',',expand=True).stack().reset_index(0)
    cell2spot_tag = dict(zip(mapping[0],mapping['level_0']))

    spot_tag2xy =dict(zip(st_simu.obs_names, [f'{x}_{y}' for x,y in xy],))
    cell2xy = {cell:spot_tag2xy[spot_tag] for cell,spot_tag in cell2spot_tag.items()}

    sc_simu = sc_data[sc_data.obs_names.isin(cell2xy)]
    sc_simu.obs['cell2spot_tag'] = sc_simu.obs_names.map(cell2spot_tag)
    sc_simu.obs['cell2xy'] = sc_simu.obs_names.map(cell2xy)

    # Write simulation h5ad files
    st_simu.write_h5ad('st_simu.h5ad')
    sc_simu.write_h5ad('sc_simu.h5ad') 

def config_plotting():
    ## %  config plotting
    sc.settings._vector_friendly = True
    # p9.theme_set(p9.theme_classic)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["savefig.transparent"] = True
    plt.rcParams["figure.figsize"] = (4, 4)
        
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.titleweight"] = 500
    plt.rcParams["axes.titlepad"] = 8.0
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = 500
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.labelpad"] = 6.0
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    plt.rcParams["font.size"] = 11
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', "Computer Modern Sans Serif", "DejaVU Sans"]
    plt.rcParams['font.weight'] = 500

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['xtick.minor.size'] = 1.375
    plt.rcParams['xtick.major.size'] = 2.75
    plt.rcParams['xtick.major.pad'] = 2
    plt.rcParams['xtick.minor.pad'] = 2

    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['ytick.minor.size'] = 1.375
    plt.rcParams['ytick.major.size'] = 2.75
    plt.rcParams['ytick.major.pad'] = 2
    plt.rcParams['ytick.minor.pad'] = 2

    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams['legend.handlelength'] = 1.4
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.scatterpoints'] = 3

    plt.rcParams['lines.linewidth'] = 1.7
    DPI = 300 # dots per inch

def spatial_plots(st_data):
    # Show underlying image (None)
    sc.pl.spatial(st_data, alpha=0, img=None, scale_factor=1, spot_size=1)

    # Show embedded spots on top of underlying image
    sc.pl.embedding(st_data, 'spatial')

    # Show embedded spots with color determined by pi value
    # Temperature = 1
    sc.pl.embedding(st_data, 'spatial', color=[x for x in st_data.obsm['pi_cell1'].columns], cmap='RdBu_r',show=False)
    plt.suptitle("$T=1$")
    plt.show()
    # Temperature = 5
    sc.pl.embedding(st_data, 'spatial', color=[x+'_5' for x in st_data.obsm['pi_cell5'].columns], cmap='RdBu_r',show=False)
    plt.suptitle("$T=5$")
    plt.show()

def plot_spatial_correlated_distribution(st_data):
    config_plotting()
    spatial_plots(st_data)

# Simulate spatial expression data with Guassian process   
def simulate_spatial_expression_data(path, n_bins, n_sc):
    # Load single cell gene expression data
    sc_data = load_sc_data(path)

    # Determine underlying spatial grid
    st_data, n_spots = generate_empty_spatial_image(n_bins, n_sc, sc_data)
    K, xy = distance_matrix(st_data)
    # plot_st_data(st_data)
    
    # Get GP samples
    gp_samples = get_gaussian_process_samples(K, n_bins, n_spots)

    # Set temperature using GP samples
    st_data = set_temperature(st_data, gp_samples, n_bins, n_sc)

    cell_types = determine_cell_groups(st_data, sc_data)

    # Calculate phi and pi values for each spot
    synthesize_data(cell_types, st_data, sc_data, xy, n_bins, n_spots)

    # Plot the resulting graph
    plot_spatial_correlated_distribution(st_data)

##############################################
# main
##############################################


def run_spatial(path, n_bins, n_sc):
 
    print("--------------------------------------")
    print("   Determining spatial coordinates    ")
    print("--------------------------------------")

    simulate_spatial_expression_data(path, n_bins, n_sc)
    

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
    adata = make_h5ad(gene_expr, ad_path, n_sc)

    return n_genes, n_bins, n_sc
