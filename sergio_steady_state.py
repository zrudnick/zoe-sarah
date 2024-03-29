
from sergio import *

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

def steady_state_technical_no_noise(sim, expr, n_genes, n_master_regs):
    # keep the code without noise
    # Convert to UMI count
    gene_expr = sim.convert_to_UMIcounts(expr)
    # Make a 2d gene expression matrix
    gene_expr = np.concatenate(gene_expr, axis = 1)

    return gene_expr

def add_dummy_counts(gene_expr, n_master_regs):
    dummy_i = n_master_regs//2
    for i in range(n_master_regs//2):
        gene_expr[i] += gene_expr[dummy_i]
        gene_expr = np.delete(gene_expr, dummy_i, axis=0)