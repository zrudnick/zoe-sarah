
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc

from scipy.spatial.distance import pdist, squareform
from tqdm import trange
from h5ad import *

##############################################
# Spatial Plotting
##############################################

# Load gene expression data
def load_sc_data(path):
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
def generate_empty_spatial_image(n_genes, n_bins, n_sc, sc_data, lr):

    # Define dimensions of the empty Visium spatial image
    n_genes = n_genes
    n_ligand = n_sc // lr
    n_spots = n_bins * (n_sc + n_ligand)

    spatial_coordinates = get_spatial_coordinates(n_spots)

    # Create an AnnData object for spatial transcriptomics data
    obs_index = [str(i) for i in np.arange(n_spots)]
    var_index = [str(i) for i in np.arange(n_genes)]
    st_data = sc.AnnData(
        X=np.zeros((n_spots, n_genes)),  # placeholder for gene expression data
        obs=pd.DataFrame(index=obs_index),  # observation metadata
        var=pd.DataFrame(index=var_index)  # variable (gene) metadata
    )
    st_data.obsm['spatial'] = np.asarray(spatial_coordinates) # spatial coordinates

    return st_data, n_spots

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

# Use K as the Kernel matrix of a GP to generate samples for each cell type
def get_gaussian_process_samples(K, n_bins, n_spots):
    mean = [0]*n_spots
    cov = K
    # gp stands for Gaussian Process
    gp_samples = np.random.multivariate_normal(mean, cov, size=(n_bins,)).T # gp sample for each spot
    return gp_samples

# Set the temperatures (T = 1 and T = 5)
def set_temperature(st_data, gp_samples, n_bins, n_sc):
    # T is the temperature
    # A small value of T tends to preserve the dominant cell type with the highest energy
    # while a large value of T maintains the original cell type proportions
 
    cell_abundance = n_sc / (n_sc * n_bins)
   
    # try T = 1 and T = 5
    for T in [1, 5]:
        # the cell type proportion at every spot is an 'energy' Phi
        columns = [str(col) for col in range(n_bins)]
        index = st_data.obs.index
        phi_cell = pd.DataFrame(gp_samples, columns=columns, index=index)
        # for each spot and each cell type:
        # each Phi is a cell type energy vector aligned with spots
        # the proportion Pi is then calculated using the energy
        pi_cell = (cell_abundance * np.exp(phi_cell/T)).div((cell_abundance * np.exp(phi_cell/T)).sum(1),axis='index')
        st_data.obsm['pi_cell' + str(T)] = pi_cell # add to existing data

        for cell_type, pi in pi_cell.items():
            st_data.obs[str(cell_type) + '_' + str(T)] = pi_cell[cell_type]
    return st_data, cell_abundance

# Format gene expression data by cell groups
def determine_cell_groups(st_data, sc_data):

    # Queues of each cell type
    celltype_order = st_data.obsm['pi_cell1'].columns.tolist() # list of cell types
    cell_groups = [x for x in sc_data.to_df().groupby(sc_data.obs['Cell Type'],sort=False)] 
    cell_groups.sort(key=lambda x: celltype_order.index(str(x[0])),)
    cell_groups = [(ct,cell_group.sample(frac=1)) for ct, cell_group in cell_groups]
    return cell_groups

# Sample a cell type
def sample_cell_type(cell_groups, cell_type_index, n):
    cell_type, type_df = cell_groups[cell_type_index]
    pop_df = type_df.iloc[:n]
    type_df.drop(pop_df.index, inplace=True)
    type_tags = pop_df.index.tolist()
    type_sum = pop_df.sum(0)
    
    # if len(type_tags) == 0:
    #     print(f'Warning: {cell_type} has no more cells')
    
    return n, type_tags, [str(cell_type)] * n, type_sum

# Sample the previous seen cell type's corresponding ligand producing cell
def sample_ligand_cell(cell_groups, cell_type_index, n_bins, n):
    receptor_cell_type, _ = cell_groups[cell_type_index]
    cell_type = receptor_cell_type + n_bins
    
    type_tags = [str(cell_type)]
    type_sum = -1
    
    return n, type_tags, [str(cell_type)] * n, type_sum

# Synthesize the data spot by spot
def determine_spot_cell_types(cell_groups, st_data, sc_data, xy, n_bins, n_spots, lr):

    # Create AnnData objects for simulation information
    st_simu = sc.AnnData(np.zeros((st_data.shape[0],sc_data.shape[1])),obs=pd.DataFrame(index=st_data.obs.index,columns=['cell_counts','cell_tags','cell_types']))
    cell_type_index = None

    # Choose cell type for each spot
    for i in trange(n_spots):
        spot_size_true = 0
        spot_tags = []
        spot_types = []
        spot_X = np.zeros(sc_data.shape[1])
        spot_size = 1

        if (i % (lr + 1) == 1): # if a ligand should be placed
            spot_size_ct, type_tags, type_list, type_sum = sample_ligand_cell(cell_groups, cell_type_index, n_bins, spot_size)
        else:
            prob_in_spot = st_data.obsm["pi_cell5"].iloc[i].values # put cells with similar receptor expression together
            choice = np.random.choice(n_bins, spot_size, p=prob_in_spot)
            cell_type_index = choice[0]
            spot_size_ct, type_tags, type_list, type_sum = sample_cell_type(cell_groups, cell_type_index, spot_size)
            
        spot_size_true += spot_size_ct
        spot_tags.extend(type_tags)
        spot_types.extend(type_list)
            
        spot_X += type_sum 
        
        st_simu.obs.iloc[i]['cell_counts'] = spot_size_true
        st_simu.obs.iloc[i]['cell_tags'] = ','.join(spot_tags)
        st_simu.obs.iloc[i]['cell_types'] = ','.join(spot_types)
        st_simu.X[i] = spot_X

    print(st_simu.X)

    st_simu.obsm['spatial'] = st_data.obsm['spatial']
    st_simu.obs['cell_counts'] = st_simu.obs['cell_counts'].astype(str)
    st_simu.var_names = sc_data.var_names

    mapping = st_simu.obs['cell_tags'].str.split(',',expand=True).stack().reset_index(0)
    cell2spot_tag = dict(zip(mapping[0],mapping['level_0']))

    spot_tag2xy =dict(zip(st_simu.obs_names, [f'{x}_{y}' for x,y in xy],))
    cell2xy = {cell:spot_tag2xy[spot_tag] for cell,spot_tag in cell2spot_tag.items()}

    sc_simu = sc_data[sc_data.obs_names.isin(cell2xy)].copy()
    sc_simu.obs['cell2spot_tag'] = sc_simu.obs_names.map(cell2spot_tag)
    sc_simu.obs['cell2xy'] = sc_simu.obs_names.map(cell2xy)

    # Write simulation h5ad files
    st_simu.write_h5ad('st_simu.h5ad')
    sc_simu.write_h5ad('sc_simu.h5ad') 

    return st_simu, sc_simu

    # to verify results:
    # plot expression of particular genes in actual grid
    # plot downstream genes

    # to add ligand:
    # replace or do while simulating data?
    # 1:n ligand expression (start with n=1)

# Configure plots
def configure_plots():
    sc.settings._vector_friendly = True
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
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

def plot_pi_values(st_data):
    # Temperature = 1
    sc.pl.embedding(st_data, 'spatial', color=[x + '_1' for x in st_data.obsm['pi_cell1'].columns], cmap='RdBu_r', show=False)
    plt.suptitle("Temperature = 1")
    plt.show()

    # Temperature = 5
    sc.pl.embedding(st_data, 'spatial', color=[x + '_5' for x in st_data.obsm['pi_cell5'].columns], cmap='RdBu_r', show=False)
    plt.suptitle("Temperature = 5")
    plt.show()

    # sc.pl.embedding(sc_simu, 'spatial', color=[x + '' for x in sc_simu.obs['cell2xy']], cmap='RdBu_r', show=False)
    # plt.show()

def plot_cell_types(st_simu, st_data):
    pi_cell_discrete = st_simu.obs['cell_types'].str.split(',',expand=True).apply(pd.value_counts,axis=1)
    
    st_data.obsm['pi_cell_discrete'] = pi_cell_discrete
    for ct,pi in pi_cell_discrete.items():
        st_data.obs[ct+'_discrete'] = pi_cell_discrete[ct]
    sc.pl.embedding(st_data, 'spatial', color=[x+'_discrete' for x in st_data.obsm['pi_cell_discrete'].columns], cmap='RdBu_r',show=False)
    plt.suptitle("Discrete T=1")
    plt.show()

    # T = 5
    pi_cell_discrete = st_simu.obs['cell_types'].str.split(',',expand=True).apply(pd.value_counts,axis=1)
    
    st_data.obsm['pi_cell_discrete'] = pi_cell_discrete
    for ct,pi in pi_cell_discrete.items():
        st_data.obs[ct+'_discrete'] = pi_cell_discrete[ct]

    sc.pl.embedding(st_data, 'spatial', color=[x+'_discrete' for x in st_data.obsm['pi_cell_discrete'].columns], cmap='RdBu_r',show=False)
    plt.suptitle("Discrete T=5")
    plt.show()

# Plot the spatial data for T = 1 and T = 5
def plot_spatial_data(st_data, st_simu, sc_simu, cell_abundance):

    # Configure plots
    configure_plots() 

    # Leave commented to omit usage of underlying sample image
    #sc.pl.spatial(st_data, alpha=0, img=None, scale_factor=1, spot_size=1)

    # Show embedded spots with color determined by Pi value
    plot_pi_values(st_data)

    # Show embedded spots with color determined by cell
    #plot_cell_types(st_simu, st_data)

# Simulate spatial expression data with Guassian process   
def simulate_spatial_expression_data(path, n_genes, n_bins, n_sc, lr):
    # Load single cell gene expression data
    sc_data = load_sc_data(path)

    # Determine underlying spatial grid
    st_data, n_spots = generate_empty_spatial_image(n_genes, n_bins, n_sc, sc_data, lr)
    K, xy = distance_matrix(st_data)
    
    # Get GP samples
    gp_samples = get_gaussian_process_samples(K, n_bins, n_spots)

    # Set temperature using GP samples
    st_data, cell_abundance = set_temperature(st_data, gp_samples, n_bins, n_sc)

    # Format cell groups
    cell_groups = determine_cell_groups(st_data, sc_data)

    # Calculate Phi and Pi values for each spot
    st_simu, sc_simu = determine_spot_cell_types(cell_groups, st_data, sc_data, xy, n_bins, n_spots, lr)

    # Plot the resulting graph
    plot_spatial_data(st_data, st_simu, sc_simu, cell_abundance)