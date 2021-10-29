import matplotlib.pyplot as plt
import numpy as np
import os

from data_process import binarize_cluster_dict, binary_cluster_dict_to_memories, parse_exptdata, load_cluster_dict, \
                         load_memories_genes_clusters
from data_settings import DATADIR, OUTPUTDIR
from data_standardize import load_npz_of_arr_genes_cells
from utils.file_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_functions import hamiltonian, hamming, single_memory_projection
from singlecell.singlecell_linalg import memory_corr_matrix_and_inv, interaction_matrix, predictivity_matrix
from singlecell.singlecell_simulate import singlecell_sim

"""
Script to check whether each cluster is representative of the basin of attraction in hopfield/projection model
 - build interaction matrix from reduced memory matrix
 - load reduced cluster data
 - get basin score for each cluster by checking if all cells (in that cluster) stay within that basin dynamically
 - plot scores 
"""

# TODO test and optimize build_basin_states
# TODO build remaining functions + unit tests

ANALYSIS_SUBDIR = "basinscores"


def is_energy_increase(intxn_matrix, data_vec_a, data_vec_b):
    # state b in basin if the energy from a to b increases AND a is in basin
    energy_a = hamiltonian(data_vec_a, intxn_matrix=intxn_matrix)
    energy_b = hamiltonian(data_vec_b, intxn_matrix=intxn_matrix)
    if energy_b > energy_a:
        return True
    else:
        return False


def build_basin_states(intxn_matrix, memory_vec,
                       recurse_dist_d=0, recurse_basin_set=None, recurse_state=None,
                       sites_flipped_already=None):
    """
    Args:
        - intxn_matrix: J_ij built from memory_matrix
        - memory_vec: column of the memory matrix
        - various recursive arguments
    Returns:
        - basin_set: dict of {num_flips: SET (not list) of states as Nx1 lists} comprising the basin
    """
    num_genes = intxn_matrix.shape[0]

    if recurse_basin_set is None:
        memory_vec_copy = np.array(memory_vec[:])
        recurse_basin_set = {d: set() for d in range(num_genes + 1)}
        recurse_basin_set[0].add(tuple(memory_vec_copy))
        recurse_state = memory_vec_copy
        sites_flipped_already = []
        recurse_dist_d = 1

    #size_basin_at_dist_d = len(recurse_basin_set[recurse_dist_d])    # number of states with hamming dist = d in the basin

    for site_idx in [val for val in range(num_genes) if val not in sites_flipped_already]:
        recurse_state_flipped = np.array(recurse_state[:])
        recurse_state_flipped[site_idx] = -1 * recurse_state[site_idx]
        if is_energy_increase(intxn_matrix, recurse_state, recurse_state_flipped):
            recurse_basin_set[recurse_dist_d].add(tuple(recurse_state_flipped))
            recurse_sites_flipped_already = sites_flipped_already[:]
            recurse_sites_flipped_already.append(site_idx)
            build_basin_states(intxn_matrix, memory_vec,
                               recurse_dist_d=recurse_dist_d + 1,
                               recurse_basin_set=recurse_basin_set,
                               recurse_state=recurse_state_flipped,
                               sites_flipped_already=recurse_sites_flipped_already)
        else:
            return recurse_basin_set

    return recurse_basin_set


def basin_projection_timeseries(k, memory_array, intxn_matrix, eta, basin_data_k, plot_data_folder,
                                num_steps=100, plot=True, flag_write=False):
    # TODO check vs analysis basin grid functions, possible overlap
    def get_memory_proj_timeseries(state_array, memory_idx):
        num_steps = np.shape(state_array)[1]
        timeseries = np.zeros(num_steps)
        for time_idx in range(num_steps):
            timeseries[time_idx] = single_memory_projection(state_array, time_idx, memory_idx, eta=eta)
        return timeseries

    TEMP = 1e-2

    proj_timeseries_array = np.zeros((num_steps, basin_data_k.shape[1]))

    for idx in range(basin_data_k.shape[1]):
        init_cond = basin_data_k[:, idx]
        cellstate_array, io_dict = singlecell_sim(
            init_state=init_cond, iterations=num_steps, beta=1/TEMP, xi=memory_array, intxn_matrix=intxn_matrix,
            memory_labels=list(range(memory_array.shape[1])), gene_labels=list(range(memory_array.shape[0])),
            flag_write=flag_write, analysis_subdir=ANALYSIS_SUBDIR, plot_period=num_steps * 2, verbose=False)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, k)[:]
    if plot:
        plt.plot(range(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
        plt.title('Test memory %d projection for all cluster member' % k)
        plt.ylabel('proj on memory %d' % (k))
        plt.xlabel('Time (%d updates, all spins)' % num_steps)
        plt.savefig(plot_data_folder + os.sep + 'cluster_%d.png' % k)
        plt.clf()
    return proj_timeseries_array


def get_basins_scores(memory_array, binarized_cluster_dict, basinscore_method="default"):
    """
    Args:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - basinscore_method: options for different basin scoring algos
                             (one based on crawling the basin exactly, other via low temp dynamics)
    Returns:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    """
    assert basinscore_method in ['crawler', 'trajectories']
    num_genes, num_clusters = memory_array.shape
    cells_each_cluster = [binarized_cluster_dict[idx].shape[1] for idx in range(num_clusters)]
    num_cells = np.sum(cells_each_cluster)
    print("num_genes, num_clusters, num_cells:\n%d %d %d" % (num_genes, num_clusters, num_cells))

    def basin_score_pairwise(basin_k, memory_vector, data_vector):
        # OPTION 1 -- is cell in basin yes/no
        # OPTION 2 -- is cell in basin - some scalar value e.g. projection onto that mem
        # OPTION 3 -- based on compare if data vec in set of basin states (from aux fn)
        hd = hamming(memory_vector, data_vector)
        if tuple(data_vector) in basin_k[hd]:
            print("data_vector in basin_k[hd]")
            return 1.0
        else:
            print("data_vector NOT in basin_k[hd]")
            return 0.0

    # 1 is build J_ij from Xi
    _, a_inv_arr = memory_corr_matrix_and_inv(memory_array, check_invertible=True)
    eta = predictivity_matrix(memory_array, a_inv_arr)
    intxn_matrix = interaction_matrix(memory_array, a_inv_arr, "projection")

    # 2 is score each cell in each cluster based on method
    score_dict = {k: 0 for k in range(num_clusters)}

    # setup io
    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)

    if basinscore_method == 'crawler':
        for k in range(num_clusters):
            print("Scoring basin for cluster:", k)
            binary_cluster_data = binarized_cluster_dict[k]
            memory_k = memory_array[:,k]
            basin_k = build_basin_states(intxn_matrix, memory_k)
            for cell_data in binary_cluster_data.T:  # TODO make sure his gives columns (len is N)
                print(len(cell_data), num_genes, cell_data.shape)
                score_dict[k] += basin_score_pairwise(basin_k, memory_k, cell_data)
            print(score_dict)
    else:
        assert basinscore_method == 'trajectories'
        for k in range(num_clusters):
            print("Scoring basin for cluster:", k)
            #init_conds = binarized_cluster_dict[k]
            print("WARNING: only looking at first 10 cells in each cluster")
            init_conds = binarized_cluster_dict[k][:,0:10]
            trajectories = basin_projection_timeseries(k, memory_array, intxn_matrix, eta, init_conds, io_dict['plotdir'],
                                                       num_steps=3, plot=True, flag_write=False)
            print(trajectories)
            score_dict[k] = np.mean(trajectories[-1,:])
    # save to file
    scores = [score_dict[k] for k in range(num_clusters)]
    np.savetxt(data_folder + os.sep + "scores.txt", scores)
    return score_dict, io_dict


def plot_basins_scores(score_dict, savedir=None):
    """
    Args:
        - score_dict: {k: M x 1 array for k in 0 ... K-1 (i.e. cluster index)}
    Returns:
        - plot axis
    """
    # 1 is build J_ij from Xi
    # 2 is score each cell in each cluster based on method
    # 3 id store scores in score_dict and return
    num_clusters = np.max(list(score_dict.keys()))
    x_axis = list(range(num_clusters))
    y_axis = [score_dict[k] for k in range(num_clusters)]
    plt.bar(x_axis, y_axis, width=0.5)
    plt.title('Basin scores for each cluster')
    plt.xlabel('cluster idx')
    plt.ylabel('basin score')
    if savedir is not None:
        plt.savefig(savedir + os.sep + 'basin_scores.pdf')
    else:
        plt.show()
    return plt.gca()


if __name__ == '__main__':
    datadir = DATADIR + os.sep + "2018_scMCA"

    # run flags
    flag_gen_basinscore = False
    flag_plot_basinscore = True
    switch_generate_from_orig_npz = False  # False is default

    # options
    basinscore_method = "trajectories"  # either 'trajectories', 'crawler'

    if flag_gen_basinscore:
        if switch_generate_from_orig_npz:
            # generation options
            verbose = True
            binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
            memory_method = "default"
            # (1) load pruned raw data
            rawpruned_path = datadir + os.sep + 'arr_genes_cells_withcluster_compressed_pruned.npz'
            arr, genes, cells = load_npz_of_arr_genes_cells(rawpruned_path, verbose=True)
            # (2) create pruned cluster dict
            cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose)
            # (3) binarize cluster dict
            binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method)
            # (4) create memory matrix
            #     - alternative: load memory array from file and assert gene lists same for example as QC check
            memory_array = binary_cluster_dict_to_memories(binarized_cluster_dict, genes, memory_method=memory_method)
        else:
            # (1) load cluster dict from file (pruned, boolean)
            cdnpz = datadir + os.sep + 'clusterdict_boolean_compressed_pruned.npz'
            binarized_cluster_dict = load_cluster_dict(cdnpz)
            # (2) load memory_array from file (pruned, boolean)
            memnpz = datadir + os.sep + 'mems_genes_types_compressed_pruned.npz'
            memory_array, genes, clusters = load_memories_genes_clusters(memnpz)
            # check size matches
            assert len(clusters) == len(list(binarized_cluster_dict.keys()))
            assert memory_array.shape[0] == binarized_cluster_dict[0].shape[0]
        # basin scores
        basin_scores, io_dict = get_basins_scores(memory_array, binarized_cluster_dict, basinscore_method=basinscore_method)

    if flag_plot_basinscore:
        if not flag_gen_basinscore:
            scorepath = OUTPUTDIR + os.sep + "scores.txt"
            basin_score_txt = np.loadtxt(scorepath)
            basin_scores = {k:float(basin_score_txt[k]) for k in range(len(basin_score_txt))}
            data_folder = OUTPUTDIR
        plot_basins_scores(basin_scores, data_folder)
