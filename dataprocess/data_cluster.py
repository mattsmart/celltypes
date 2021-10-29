import numpy as np
import os

from data_settings import DATADIR
from data_standardize import load_npz_of_arr_genes_cells

"""
Script to load raw data and attach cluster labels
 - if no cluster labels provide, perform clustering (TODO: implement)
 - else append saved clusters
"""


def load_cluster_labels(clusterpath, one_indexed=True):
    """
    one_indexed: if true, assume cluster index starts at 1 (e.g. as in 2018_scMCA data)
    """
    cluster_labels = {}
    if one_indexed:
        dec = 1
    else:
        dec = 0
    # expected format of csv file is "cluster number, name"
    with open(clusterpath) as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            line = line.split(',')
            print(line)
            cluster_labels[int(line[0])-dec] = line[1]
    return cluster_labels


def attach_cluster_id_arr_manual(npzpath, clusterpath, save=True, one_indexed=True):
    """
    one_indexed: if true, assume cluster index starts at 1 (as in scMCA)
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    # generate cell_to_cluster_idx mapping
    cluster_info = {}
    # expected format of csv file is "cell name, cluster idx, tissue origin"
    with open(clusterpath) as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            line = line.split(',')
            cluster_info[line[0]] = int(line[1])
    # adjust genes and arr contents
    arr = np.insert(arr, 0, 0, axis=0)
    genes = np.insert(genes, 0, 'cluster_id')  # TODO should have global constant for this mock gene label
    if one_indexed:
        for idx in range(len(cells)):
            arr[0,idx] = cluster_info[cells[idx]] - 1
    else:
        for idx in range(len(cells)):
            arr[0,idx] = cluster_info[cells[idx]]
    # save and return data
    if save:
        print("saving cluster-appended arrays...")
        datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
        np.savez_compressed(datadir + os.sep + "arr_genes_cells_withcluster_compressed.npz", arr=arr, genes=genes, cells=cells)
    return arr, genes, cells


if __name__ == '__main__':
    datadir = DATADIR
    flag_attach_clusters_resave = False

    if flag_attach_clusters_resave:
        compressed_file = datadir + os.sep + "arr_genes_cells_raw_compressed.npz"
        clusterpath = datadir + os.sep + "SI_cells_to_clusters.csv"
        arr, genes, cells = attach_cluster_id_arr_manual(compressed_file, clusterpath, save=True)
