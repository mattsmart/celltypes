import numpy as np
import os

from data_settings import DATADIR
from data_standardize import load_npz_of_arr_genes_cells

"""
Script to reduce row count (number of genes) in single-cell RNA expression data (N genes x M cells)
- prune boring rows: delete rows which are all on or all off
- prune duplicate rows: delete rows which are copies of other rows
TODO: less naive dimension reduction (PCA, others)
"""


def prune_rows(npzpath, specified_rows=None, save_pruned=True, save_rows=True, del_A=True, del_B=True):
    """
    Delete rows from array and corresponding genes that are self-duplicates
    NOTE: very similar to reduce_gene_set(xi, gene_labels)
    """
    arr, genes, cells = load_npz_of_arr_genes_cells(npzpath)
    num_rows, num_cols = arr.shape
    print("CHECK FIRST ROW NOT CLUSTER ROW:", genes[0])
    if specified_rows is None:
        # collect rows to delete (A - self-duplicate rows all on / all off)
        if del_A:
            rows_duplicates = np.all(arr.T == arr.T[0,:], axis=0)
            rows_to_delete_self_dup = set([idx for idx, val in enumerate(rows_duplicates) if val])
            print("number of self-duplicate rows:", len(rows_to_delete_self_dup))

        # collect rows to delete (B - rows which are copies of other rows)
        if del_B:
            _, unique_indices = np.unique(arr, return_index=True, axis=0)
            rows_to_delete_dupe = set(range(num_rows)) - set(unique_indices)
            print("number of duplicated rows (num to delete):", len(rows_to_delete_dupe))

        # prepare rows to delete based on deletion choice
        if del_A and del_B:
            rows_to_delete = np.array(list(rows_to_delete_dupe.union(rows_to_delete_self_dup)))
        elif del_A:
            rows_to_delete = np.array(list(rows_to_delete_self_dup))
        else:
            rows_to_delete = np.array(list(rows_to_delete_dupe))
    else:
        save_rows = False
        rows_to_delete = np.array(specified_rows)
    # adjust genes and arr contents
    print("Orig shape arr, genes, cells:", arr.shape, genes.shape, cells.shape)
    arr = np.delete(arr, rows_to_delete, axis=0)
    genes = np.delete(genes, rows_to_delete)  # TODO should have global constant for this mock gene label
    print("New shape arr, genes, cells:", arr.shape, genes.shape, cells.shape)
    # save and return data
    datadir = os.path.abspath(os.path.join(npzpath, os.pardir))
    if save_pruned:
        print("saving pruned arrays...")
        base = os.path.basename(npzpath)
        basestr = os.path.splitext(base)[0]
        savestr = basestr + '_pruned.npz'
        np.savez_compressed(datadir + os.sep + savestr, arr=arr, genes=genes, cells=cells)
    if save_rows:
        np.savetxt(datadir + os.sep + 'rows_to_delete.txt', rows_to_delete, delimiter=",", fmt="%d")
    return rows_to_delete, arr, genes, cells


def reduce_gene_set(xi, gene_labels):  # TODO: my removal ends with 1339 left but theirs with 1337 why?
    """
    NOTE: very similar to prune_boring_rows(...)
    """
    genes_to_remove = []
    for row_idx, row in enumerate(xi):
        if all([x == row[0] for x in row]):
            genes_to_remove.append(row_idx)
    reduced_gene_labels = [gene_labels[idx] for idx in range(len(xi)) if idx not in genes_to_remove]
    reduced_xi = np.array([row for idx, row in enumerate(xi) if idx not in genes_to_remove])
    return reduced_gene_labels, reduced_xi


if __name__ == '__main__':
    datadir = DATADIR
    flag_prune_rows = False
    flag_prune_duplicate_rows = False
    flag_create_and_apply_rows_to_delete = True

    if flag_prune_rows:
        compressed_file = datadir + os.sep + "mems_genes_types_compressed.npz"
        prune_rows(compressed_file)

    if flag_create_and_apply_rows_to_delete:
        genes_to_keep_file = 'genes_to_keep_pruned_mouse_TF.txt'
        mems_file_to_reduce = 'mems_genes_types_compressed_pruned_A.npz'
        # load genes_to_keep_file
        with open(genes_to_keep_file, 'r') as in_file:
            genes_to_keep = in_file.read().split('\n')
        # load arr genes cells
        arr, genes, cells = load_npz_of_arr_genes_cells(mems_file_to_reduce)
        # create list of indices to delete by comparing names
        non_TF_rows = []
        for idx, gene in enumerate(genes):
            if gene not in genes_to_keep:
                non_TF_rows.append(idx)
        # select those indices from the numpy array
        prune_rows(mems_file_to_reduce, specified_rows=non_TF_rows, save_pruned=True, save_rows=True)
