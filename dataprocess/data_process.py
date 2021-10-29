import numpy as np
import os

from data_cluster import load_cluster_labels, attach_cluster_id_arr_manual
from data_rowreduce import prune_rows
from data_settings import DATADIR, PIPELINES_VALID, CELLSTOCLUSTERS_2018SCMCA, NPZ_2018SCMCA_MEMS, NPZ_2018SCMCA_ORIG, \
                          NPZ_2018SCMCA_ORIG_WITHCLUSTER, NPZ_2014MEHTA_ORIG, PIPELINES_DIRS
from data_standardize import save_npz_of_arr_genes_cells, load_npz_of_arr_genes_cells


"""
Purpose: process standardized expression data (i.e. converted to npz of arr, genes, cells)
 - cluster data, or load in clustered results and attach it to first row of gene, expression in the npz
 - save clustered raw data in standard npz format (npz of arr, genes, cells)
 - convert raw data into "cluster dict": dictionary that maps cluster data to submatrix of genes x cells
 - binarize data within each cluster dict
 - create binarized cluster dict
 - from binarized cluster dict: create "memory" / "cell type" matrix (get representative column from each cluster)
 - save memory matrix in standard npz format (npz of mems, genes, types)
 - reduce row number with various pruning techniques
 - save total row reduction in file "removed_rows.txt"
 - save reduced memory matrix in standard npz format (npz of mems, genes, types)
 - use "removed_rows.txt" to delete rows of original raw data 
 - save reduced clustered raw data in standard npz format (npz of arr, genes, cells)
 - save reduced unclustered raw data in standard npz format (npz of arr, genes, cells)
Main output:
 - reduced memory matrix is used as input to singlecell module
"""

# TODO pass metadata to all functions?
# TODO test and optimize build_basin_states
# TODO build remaining functions + unit tests
# TODO have report script which stores all processing flags/choices/order
# TODO maybe have rundir for results of each proc run
# TODO how to save cluster dict? as npz?


def binarize_data(xi):
    return 1.0 * np.where(xi > 0, 1, -1)  # mult by 1.0 to cast as float


def binarize_cluster_dict(cluster_dict, metadata, binarize_method="by_gene", savedir=None):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - binarize_method: options for different binarization methods: by_cluster or by_gene (default)
        - savedir: dir to save cluster_dict
    Returns:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
    """
    assert binarize_method in ['by_cluster', 'by_gene']
    num_clusters = metadata['num_clusters']

    print(num_clusters, np.max(list(cluster_dict.keys())), cluster_dict[0].shape)

    binarize_cluster_dict = {}
    if binarize_method == 'by_gene':
        for k in range(num_clusters):
            cluster_data = cluster_dict[k]
            min_gene_vals = np.amin(cluster_data, axis=1)  # min value each gene has over all cells in the cluster
            max_gene_vals = np.amax(cluster_data, axis=1)
            mids = 0.5 * (min_gene_vals - max_gene_vals)
            # TODO vectorize this
            binarized_cluster = np.zeros(cluster_data.shape, dtype=np.int8)
            for idx in range(cluster_data.shape[0]):
                binarized_cluster[idx,:] = np.where(cluster_data[idx,:] > mids[idx], 1.0, -1.0)  # mult by 1.0 to cast as float
            binarize_cluster_dict[k] = binarized_cluster
    else:
        print("WARNING: binarize_method by_cluster is not stable (data too sparse)")
        for k in range(num_clusters):
            cluster_data = cluster_dict[k]
            min_val = np.min(cluster_data)
            max_val = np.max(cluster_data)
            mid = 0.5 * (max_val - min_val)
            binarized_cluster = 1.0 * np.where(cluster_data > mid, 1, -1)  # mult by 1.0 to cast as float
            binarized_cluster.astype(np.int8)
            binarize_cluster_dict[k] = binarized_cluster

    # save cluster_dict
    if savedir is not None:
        cdnpz = savedir + os.sep + 'clusterdict_boolean_compressed.npz'
        save_cluster_dict(cdnpz, binarize_cluster_dict)
    return binarize_cluster_dict


def binary_cluster_dict_to_memories(binarized_cluster_dict, gene_labels, memory_method="default", savedir=None):
    """
    Args:
        - binarized_cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - gene_labels: N x 1 array of 'gene_labels' for each row
        - memory_method: options for different memory processing algos
        - savedir: where to save the memory file (None -> don't save)
    Returns:
        - memory_array: i.e. xi matrix, will be N x K (one memory from each cluster)
    """
    if gene_labels[0] == 'cluster_id':
        print("Warning: gene_labels[0] == 'cluster_id', removing first element")
        gene_labels = gene_labels[1:]
    num_genes = len(gene_labels)
    num_clusters = len(list(binarized_cluster_dict.keys()))
    print("num_genes", num_genes)

    eps = 1e-4  # used to bias the np.sign(call) to be either 1 or -1 (breaks ties towards on state)
    memory_array = np.zeros((num_genes, num_clusters))
    for k in range(num_clusters):
        cluster_arr = binarized_cluster_dict[k]
        cluster_arr_rowsum = np.sum(cluster_arr, axis=1)
        memory_vec = np.sign(cluster_arr_rowsum + eps)
        memory_array[:,k] = memory_vec
    if savedir is not None:
        npzpath = savedir + os.sep + 'mems_genes_types_compressed.npz'
        store_memories_genes_clusters(npzpath, memory_array, np.array(gene_labels))
    return memory_array


def store_memories_genes_clusters(npzpath, mem_arr, genes):
    # TODO move cluster labels to metadata with gene labels, pass to this function
    cluster_id = load_cluster_labels(DATADIR + os.sep + '2018_scMCA' + os.sep + 'SI_cluster_labels.csv')
    clusters = np.array([cluster_id[idx] for idx in range(len(list(cluster_id.keys())))])
    save_npz_of_arr_genes_cells(npzpath, mem_arr, genes, clusters)
    return


def load_memories_genes_clusters(npzpath):
    mem_arr, genes, clusters = load_npz_of_arr_genes_cells(npzpath, verbose=False)
    return mem_arr, genes, clusters


def prune_memories_genes(npzpath):
    rows_to_delete, mem_arr, genes, clusters = prune_rows(npzpath, save_pruned=True, save_rows=True)
    return rows_to_delete, mem_arr, genes, clusters


def prune_cluster_dict(cluster_dict, rows_to_delete, savedir=None):
    """
    Args:
        - cluster_dict: {k: N x M array for k in 0 ... K-1 (i.e. cluster index)}
        - rows_to_delete: rows to delete from each array (val) in cluster_dict
        - savedir: where to save the memory file (None -> don't save)
    """
    pruned_cluster_dict = {k: 0 for k in list(cluster_dict.keys())}
    for k in range(len(list(cluster_dict.keys()))):
        cluster_data = cluster_dict[k]
        pruned_cluster_dict[k] = np.delete(cluster_data, rows_to_delete, axis=0)
    # save pruned_cluster_dict
    if savedir is not None:
        cdnpz = savedir + os.sep + 'clusterdict_boolean_compressed_pruned.npz'
        save_cluster_dict(cdnpz, pruned_cluster_dict)
    return pruned_cluster_dict


def save_cluster_dict(npzpath, cluster_dict):
    # convert int keys to str (and deconvert on loading)
    print("saving cluster dict at %s..." % npzpath)
    cluster_dict = {str(k):v for k,v in cluster_dict.items()}
    np.savez_compressed(npzpath, **cluster_dict)
    print("done saving cluster dict")
    return


def load_cluster_dict(npzpath):
    print("loading cluster dict at %s..." % npzpath)
    cluster_dict = np.load(npzpath)
    # convert str keys back to int
    cluster_dict = {int(k):v for k,v in cluster_dict.items()}
    print("done loading cluster dict")
    return cluster_dict


def parse_exptdata(states_raw, gene_labels, verbose=True, savedir=None):
    #TODO metadata not really used, maybe omit and rename function? else use it
    """
    Args:
        - states_raw: stores array of state data and cluster labels for each cell state (column)
        - gene_labels: stores row names i.e. gene or PCA labels (expect list or numpy array)
        - savedir: dir to save npz of clusterdict in
    Notes: data format may change with time
        - ***ASSUMED*** convention is first row stores cluster index, from 0 to np.max(row 0) == K - 1
        - future convention may be to store unique integer ID for each column corresponding to earlier in pipeline
        - maybe also extract and pass metadata_dict info (i.e. K, N, M, filename information on pipeline)
    Returns:
        - cluster_dict: {cluster_idx: N x M array of raw cell states in the cluster (i.e. not binarized)}
        - metadata: dict, mainly stores N x 1 array of 'gene_labels' for each row
    """
    if type(gene_labels) is np.ndarray:
        gene_labels = gene_labels.tolist()
    else:
        assert type(gene_labels) is list

    states_row0 = states_raw[0, :]
    states_truncated = states_raw[1:, :]
    num_genes, num_cells = states_truncated.shape  # aka N, M
    num_clusters = np.max(states_raw[0, :]) + 1
    if verbose:
        print("raw data dimension: %d x %d" % (states_raw.shape))
        print("cleaned data dimension: %d x %d" % (states_truncated.shape))
        print("num_clusters is %d" % num_clusters)

    # process gene labels
    assert len(gene_labels) == num_genes or len(gene_labels) == num_genes + 1
    if len(gene_labels) == num_genes + 1:
        assert 'cluster' in gene_labels[0]
        gene_labels = gene_labels[1:]

    # prep cluster_dict
    cluster_dict = {}
    cluster_indices = {k: [] for k in range(num_clusters)}
    # TODO optimize this chunk if needed
    for cell_idx in range(num_cells):
        cluster_idx = states_row0[cell_idx]
        cluster_indices[cluster_idx].append(cell_idx)

    # build cluster dict
    if verbose:
        print("cluster_indices collected; building cluster arrays...")
    for k in range(num_clusters):
        print(k)
        cluster_dict[k] = states_truncated.take(cluster_indices[k], axis=1)

    # fill metatadata dict
    metadata = {}
    metadata['gene_labels'] = gene_labels
    metadata['num_clusters'] = num_clusters
    metadata['K'] = num_clusters
    metadata['num_genes'] = num_genes
    metadata['N'] = num_genes
    metadata['num_cells'] = num_cells
    metadata['M'] = num_cells

    # save cluster_dict
    if savedir is not None:
        cdnpz = savedir + os.sep + 'clusterdict_compressed.npz'
        save_cluster_dict(cdnpz, cluster_dict)
    return cluster_dict, metadata


if __name__ == '__main__':

    # choose pipeline from PIPELINES_VALID
    pipeline = "2018_scMCA"
    assert pipeline in PIPELINES_VALID
    datadir = PIPELINES_DIRS[pipeline]

    if pipeline == "2014_mehta":
        flag_load_raw = True
        flag_prune_mems = True
        flag_prune_rawdata = True
        # options
        verbose = True

        # part 1: binarize standardized npz zscore data
        npzpath = NPZ_2014MEHTA_ORIG
        expression_data, genes, celltypes = load_npz_of_arr_genes_cells(npzpath, verbose=True)
        xi = binarize_data(expression_data)
        compressed_boolean = datadir + os.sep + "mems_genes_types_boolean_compressed.npz"
        save_npz_of_arr_genes_cells(compressed_boolean, xi, genes, celltypes)
        # part 3: load npz, prune, save
        rows_to_delete, xi, genes, celltypes = prune_rows(compressed_boolean, save_pruned=True, save_rows=True)

    elif pipeline == "2018_scMCA":
        flag_add_cluster_to_raw = False
        flag_build_cluster_dict_metadata = False
        flag_binarize_clusters = False
        flag_gen_memory_matrix = True
        flag_prune_mems = False
        flag_prune_raw = False
        flag_prune_binary_cluster_dict = True
        # options
        verbose = True
        binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
        memory_method = "default"

        # (1) ensure raw data npz created (using data_standardize.npz)
        # (2) cluster raw data npz (add known clusters)  -> raw npz wth cluster row
        if flag_add_cluster_to_raw:
            clusterpath = CELLSTOCLUSTERS_2018SCMCA
            arr, genes, cells = attach_cluster_id_arr_manual(NPZ_2018SCMCA_ORIG, clusterpath, save=True, one_indexed=True)
        # (3) process npz of raw data with clusters  -> cluster_dict, metadata
        if flag_build_cluster_dict_metadata:
            if not flag_add_cluster_to_raw:
                arr, genes, cells = load_npz_of_arr_genes_cells(NPZ_2018SCMCA_ORIG_WITHCLUSTER)
            cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose, savedir=datadir)

        # (4) binarize cluster dct -> binarized_cluster_dict
        if flag_binarize_clusters:
            if not flag_build_cluster_dict_metadata:
                arr, genes, cells = load_npz_of_arr_genes_cells(NPZ_2018SCMCA_ORIG_WITHCLUSTER)
                cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose, savedir=datadir)
            binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method,
                                                           savedir=datadir)
        # (5) -> boolean memory matrix, mem genes types npz
        if flag_gen_memory_matrix:
            if not flag_binarize_clusters:
                arr, genes, cells = load_npz_of_arr_genes_cells(NPZ_2018SCMCA_ORIG_WITHCLUSTER)
                cluster_dict, metadata = parse_exptdata(arr, genes, verbose=verbose, savedir=datadir)
                binarized_cluster_dict = binarize_cluster_dict(cluster_dict, metadata, binarize_method=binarize_method,
                                                               savedir=datadir)
            _ = binary_cluster_dict_to_memories(binarized_cluster_dict, genes, memory_method=memory_method,
                                                savedir=datadir)
        # (6) prune mems -> boolean memory matrix (pruned), mem genes types npz
        if flag_prune_mems:
            rawmems_npzpath = NPZ_2018SCMCA_MEMS
            rows_to_delete, memory_array, genes, clusters = prune_memories_genes(rawmems_npzpath)  # TODO prune cluster dict based on this pruning...
        # (7) prune raw data -> boolean memory matrix (pruned), mem genes types npz
        if flag_prune_raw:
            if not flag_prune_mems:
                rows_to_delete = np.loadtxt(datadir + os.sep + "rows_to_delete.txt", delimiter=',')
            rows_to_delete_increment_for_clusterrow = [i+1 for i in rows_to_delete]
            _, _, _, _ = prune_rows(NPZ_2018SCMCA_ORIG, specified_rows=rows_to_delete, save_pruned=True, save_rows=False)
            _, _, _, _ = prune_rows(NPZ_2018SCMCA_ORIG_WITHCLUSTER, specified_rows=rows_to_delete_increment_for_clusterrow,
                                    save_pruned=True, save_rows=False)
        if flag_prune_binary_cluster_dict:
            if not flag_prune_mems:
                rows_to_delete = np.loadtxt(datadir + os.sep + "rows_to_delete.txt", delimiter=',')
            cdnpz = datadir + os.sep + 'clusterdict_boolean_compressed.npz'
            binarized_cluster_dict = load_cluster_dict(cdnpz)
            _ = prune_cluster_dict(binarized_cluster_dict, rows_to_delete, savedir=datadir)
    else:
        # TODO
        flag_load_raw = True
        flag_prune_mems = True
        flag_prune_rawdata = True
        # options
        verbose = True
        binarize_method = "by_gene"  # either 'by_cluster', 'by_gene'
        memory_method = "default"
        print("No misc processing implemented")
