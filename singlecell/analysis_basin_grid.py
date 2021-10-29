import utils.init_multiprocessing  # import before numpy
import numpy as np
import os
import time
from multiprocessing import cpu_count

from singlecell.analysis_basin_plotting import plot_basin_grid, grid_video, plot_overlap_grid
from singlecell.analysis_basin_transitions import \
    ensemble_projection_timeseries, get_basin_stats, fast_basin_stats, get_init_info, \
    ANNEAL_PROTOCOL, FIELD_APPLIED_PROTOCOL, ANALYSIS_SUBDIR, SPURIOUS_LIST, OCC_THRESHOLD, \
    save_and_plot_basinstats, load_basinstats, fetch_from_run_info
from singlecell.singlecell_constants import ASYNC_BATCH, MEMS_MEHTA, MEMS_SCMCA
from singlecell.singlecell_functions import hamming
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import run_subdir_setup, runinfo_append, RUNS_FOLDER


def gen_basin_grid(ensemble, num_processes, simsetup=None, num_steps=100, anneal_protocol=ANNEAL_PROTOCOL,
                   field_protocol=FIELD_APPLIED_PROTOCOL, occ_threshold=OCC_THRESHOLD, async_batch=ASYNC_BATCH, saveall=False,
                   save=True, plot=False, verbose=False, parallel=True):
    """
    generate matrix G_ij of size p x (p + k): grid of data between 0 and 1
    each row represents one of the p encoded basins as an initial condition
    each column represents an endpoint of the simulation starting at a given basin (row)
    G_ij would represent: starting in cell type i, G_ij of the ensemble transitioned to cell type j
    """
    # simsetup unpack for labelling plots
    if simsetup is None:
        simsetup = singlecell_simsetup()
    celltype_labels = simsetup['CELLTYPE_LABELS']

    io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
    basin_grid = np.zeros((len(celltype_labels), len(celltype_labels)+len(SPURIOUS_LIST)))
    for idx, celltype in enumerate(celltype_labels):
        print("Generating row: %d, %s" % (idx, celltype))
        if saveall:
            assert parallel
            plot_all = False
            proj_timeseries_array, basin_occupancy_timeseries, _, _ = \
                ensemble_projection_timeseries(celltype, ensemble, num_proc, simsetup=simsetup, num_steps=num_steps,
                                               anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                               occ_threshold=occ_threshold, async_batch=async_batch,
                                               plot=False, output=False)
            save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                     simsetup=simsetup, prefix=celltype, occ_threshold=occ_threshold, plot=plot_all)
        else:
            init_state, init_id = get_init_info(celltype, simsetup)
            if parallel:
                transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, _ = \
                    fast_basin_stats(celltype, init_state, init_id, ensemble, num_processes, simsetup=simsetup,
                                     num_steps=num_steps, anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                     occ_threshold=occ_threshold, async_batch=async_batch, verbose=verbose)
            else:
                # Unparallelized for testing/profiling:
                transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, _ = \
                    get_basin_stats(celltype, init_state, init_id, ensemble, 0, simsetup, num_steps=num_steps,
                                    anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                    async_batch=async_batch, occ_threshold=occ_threshold, verbose=verbose)
                proj_timeseries_array = proj_timeseries_array / ensemble  # ensure normalized (get basin stats won't do this)
        # fill in row of grid data from each celltype simulation
        basin_grid[idx, :] = basin_occupancy_timeseries[:,-1]
    if save:
        np.savetxt(io_dict['latticedir'] + os.sep + 'gen_basin_grid.txt', basin_grid, delimiter=',', fmt='%.4f')
    if plot:
        plot_basin_grid(basin_grid, ensemble, num_steps, celltype_labels, io_dict['latticedir'], SPURIOUS_LIST)
    return basin_grid, io_dict


def load_basin_grid(filestr_data):
    # TODO: prepare IO functions for standardized sim settings dict struct
    basin_grid = np.loadtxt(filestr_data, delimiter=',', dtype=float)
    #sim_settings = load_sim_settings(filestr_settings)
    return basin_grid


def grid_stats(grid_data, printtorank=10):
    """
    Prints info based on row statistics of the grid
    Args:
        grid_data: basin grid data from basin occupancy hopping sim
    Returns:
        None
    """
    basin_row_sum = np.sum(grid_data, axis=1)
    ensemble = basin_row_sum[0]
    ref_list = celltype_labels + SPURIOUS_LIST
    for row in range(len(celltype_labels)):
        assert basin_row_sum[row] == ensemble  # make sure all rows sum to expected value
        sortedmems_smalltobig = np.argsort(grid_data[row, :])
        sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
        print("\nRankings for row", row, celltype_labels[row], "(sum %d)" % int(basin_row_sum[row]))
        for rank in range(printtorank):
            ranked_col_idx = sortedmems_bigtosmall[rank]
            ranked_label = ref_list[ranked_col_idx]
            print(rank, ranked_label, grid_data[row, ranked_col_idx], grid_data[row, ranked_col_idx] / ensemble)


def static_overlap_grid(simsetup, calc_hamming=False, savedata=True, plot=True):
    """
    Args:
        simsetup: project standard sim object
    Returns:
        grid_data = np.dot(xi.T, xi) -- the (p x p) correlation matrix of the memories OR
                    (p x p) array of hamming distance between all the memory pairs (equivalent; transformed)
    """
    # generate
    celltypes = simsetup["CELLTYPE_LABELS"]
    xi = simsetup["XI"]
    if calc_hamming:
        grid_data = np.zeros((len(celltypes), len(celltypes)))
        for i in range(len(celltypes)):
            for j in range(len(celltypes)):
                hd = hamming(xi[:, i], xi[:, j])
                grid_data[i, j] = hd
                grid_data[j, i] = hd
    else:
        grid_data = np.dot(xi.T, xi)
    # save and plot
    outdir = RUNS_FOLDER
    if savedata:
        dataname = 'celltypes_%s.txt' % ["overlap", "hammingdist"][calc_hamming]
        np.savetxt(outdir + os.sep + dataname, grid_data, delimiter=',', fmt='%.4f')
    if plot:
        plot_overlap_grid(grid_data, celltypes, outdir, ext='.pdf', normalize=False)
    return grid_data


def stoch_from_distance(distance_data, kappa=None):
    """
    Given a square matrix of distances between N points
    - stretch the distances according to a transformation rule (currently only np.exp(-kappa*d_ab) with kappa > 0
    - set diagonals to zero
    - normalize columns so that it represents a stochastic rate matrix (diagonals are -1*sum(col))
    - return the generated stochastic rate matrix
    """
    def transform_distance(d_ab, kappa):
        if kappa is None:
            return 1/d_ab
        else:
            return np.exp(-kappa*d_ab)
    stoch_array = np.zeros(distance_data.shape)
    for col in range(distance_data.shape[1]):
        colsum = 0.0
        for row in range(distance_data.shape[0]):
            if row != col:
                distmod = transform_distance(distance_data[row, col], kappa)
                stoch_array[row, col] = distmod
                colsum += distmod
            else:
                stoch_array[row, col] = 0.0
        stoch_array[col, col] = -colsum
        stoch_array[:, col] /= colsum  # TODO this step should not be necessary
    return stoch_array


if __name__ == '__main__':
    run_basin_grid = False
    gen_overlap_grid = False
    load_and_plot_basin_grid = False
    load_and_compare_grids = False
    reanalyze_grid_over_time = True
    make_grid_video = True
    print_grid_stats_from_file = False

    # prep simulation globals
    simsetup = singlecell_simsetup(npzpath=MEMS_MEHTA)
    celltype_labels = simsetup['CELLTYPE_LABELS']

    if run_basin_grid:
        ensemble = 1000
        timesteps = 500
        field_protocol = FIELD_APPLIED_PROTOCOL
        anneal_protocol = ANNEAL_PROTOCOL
        num_proc = cpu_count() / 2
        async_batch = True
        plot = False
        saveall = True
        parallel = True

        # run gen_basin_grid
        t0 = time.time()
        basin_grid, io_dict = gen_basin_grid(ensemble, num_proc, simsetup=simsetup, num_steps=timesteps,
                                             anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                             async_batch=async_batch, saveall=saveall, plot=plot, parallel=parallel)
        t1 = time.time() - t0
        print("GRID TIMER:", t1)

        # add info to run info file TODO maybe move this INTO the function?
        info_list = [['fncall', 'gen_basin_grid()'], ['ensemble', ensemble], ['num_steps', timesteps],
                     ['num_proc', num_proc], ['anneal_protocol', anneal_protocol], ['field_protocol', field_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['async_batch', async_batch], ['time', t1]]
        runinfo_append(io_dict, info_list, multi=True)

    if gen_overlap_grid:
        static_grid_data = static_overlap_grid(simsetup, calc_hamming=True)

    # direct data plotting
    if load_and_plot_basin_grid:
        # specify paths and load data / parameters
        groupdir = RUNS_FOLDER + os.sep + "gridmovie"
        basedirs = ['grid_781444_1kx500_2018scMCA_mir21_lvl2']
        for basedir in basedirs:
            datadir = groupdir + os.sep + basedir
            filestr_data = datadir + os.sep + "grid_at_step_499.txt"
            basin_grid_data = load_basin_grid(filestr_data)
            ensemble, num_steps = fetch_from_run_info(datadir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
            # build grid plots
            plot_basin_grid(basin_grid_data, ensemble, num_steps, celltype_labels, datadir, SPURIOUS_LIST,
                            relmax=False, ext='.pdf', vforce=0.2, namemod=basedir)
            plot_basin_grid(basin_grid_data, ensemble, num_steps, celltype_labels, datadir, SPURIOUS_LIST,
                            relmax=False, ext='.pdf', vforce=0.5, namemod=basedir)
            plot_basin_grid(basin_grid_data, ensemble, num_steps, celltype_labels, datadir, SPURIOUS_LIST,
                            relmax=False, ext='.pdf', vforce=1.0, namemod=basedir)

    # direct data plotting
    if load_and_compare_grids:
        kappa = 1.0  # positive number or None
        comparisondir = RUNS_FOLDER + os.sep + "comparegrids"
        # load basin grid
        rundir = comparisondir + os.sep + "aug11 - 1000ens x 500step"
        latticedir = rundir + os.sep + "lattice"
        filestr_data = latticedir + os.sep + "gen_basin_grid.txt"
        simulated_data = load_basin_grid(filestr_data)
        ensemble, num_steps = fetch_from_run_info(rundir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
        plot_basin_grid(simulated_data, ensemble, num_steps, celltype_labels, comparisondir, SPURIOUS_LIST,
                        relmax=False, ext='.pdf', vforce=0.5, plotname='simulated_endpt')
        simulated_data_normed = simulated_data / ensemble
        # load static distance grid
        distance_path = comparisondir + os.sep + "celltypes_hammingdist.txt"
        distance_data_normed = load_basin_grid(distance_path) / simsetup['N']  # note normalization
        plot_overlap_grid(distance_data_normed, celltype_labels, comparisondir, hamming=True,
                          relmax=True, ext='.pdf', plotname='distances_matrix')
        # transform distance grid via f(d_ab)
        stochastic_matrix = stoch_from_distance(distance_data_normed, kappa=kappa)
        stochastic_matrix_nodiag = stochastic_matrix - np.diag(np.diag(stochastic_matrix))
        plot_overlap_grid(stochastic_matrix, celltype_labels, comparisondir,
                          hamming=True, relmax=True, ext='.pdf', plotname='stochastic_matrix')
        plot_overlap_grid(stochastic_matrix_nodiag, celltype_labels, comparisondir,
                          hamming=True, relmax=True, ext='.pdf', plotname='stochastic_matrix_nodiag')  # deleted diags
        # compute deviations
        truncated_sim = simulated_data_normed[:, 0:-len(SPURIOUS_LIST)]
        print(truncated_sim.shape)
        deviation_matrix = stochastic_matrix_nodiag - truncated_sim.T
        print(simulated_data[0:4, 0:4])
        print(simulated_data_normed[0:4, 0:4])
        print(distance_data_normed[0:4, 0:4])
        print(stochastic_matrix[0:4, 0:4])
        print(stochastic_matrix_nodiag[0:4, 0:4])
        print(deviation_matrix[0:4, 0:4])
        plot_overlap_grid(deviation_matrix, celltype_labels, comparisondir,
                          hamming=True, relmax=True, ext='.pdf', plotname='deviation_matrix')
        # solve for scaling constant such that difference is minimized (A-cB=0 => AB^-1=cI)
        inverted_deviation = stochastic_matrix_nodiag*np.linalg.inv(truncated_sim.T)
        print(inverted_deviation[0:4, 0:4])

    # use labelled collection of timeseries from each row to generate multiple grids over time
    if reanalyze_grid_over_time:
        # step 0 specify ensemble, num steps, and location of row data
        groupdir = RUNS_FOLDER + os.sep + 'gridmovie'
        basedirs = ['grid_785963_1kx500_2014mehta_mir21_lvl3']
        for basedir in basedirs:
            datadir = groupdir + os.sep + basedir
            print("working in", datadir)

            ensemble, num_steps = fetch_from_run_info(datadir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
            # step 1 restructure data
            rowdatadir = datadir + os.sep + "data"
            latticedir = datadir + os.sep + "lattice"
            plotlatticedir = datadir + os.sep + "plot_lattice"
            p = len(celltype_labels)
            k = len(SPURIOUS_LIST)
            grid_over_time = np.zeros((p, p+k, num_steps))
            for idx, celltype in enumerate(celltype_labels):
                print("loading:", idx, celltype)
                proj_timeseries_array, basin_occupancy_timeseries = load_basinstats(rowdatadir, celltype)
                grid_over_time[idx, :, :] += basin_occupancy_timeseries
            # step 2 save and plot
            vforce = 0.5
            filename = 'grid_at_step'
            for step in range(num_steps):
                print("step", step)
                grid_at_step = grid_over_time[:, :, step]
                namemod = '_%d' % step
                np.savetxt(latticedir + os.sep + filename + namemod + '.txt', grid_at_step, delimiter=',', fmt='%.4f')
                plot_basin_grid(grid_at_step, ensemble, step, celltype_labels, plotlatticedir, SPURIOUS_LIST,
                                plotname=filename, relmax=False, vforce=vforce, namemod=namemod, ext='.jpg')

    if make_grid_video:
        custom_fps = 5  # 1, 5, or 20 are good
        groupdir = RUNS_FOLDER + os.sep + 'gridmovie'
        basedirs = ['grid_785963_1kx500_2014mehta_mir21_lvl3']
        for basedir in basedirs:
            datadir = groupdir + os.sep + basedir
            vidname = "%s_vmax0.5_fps%d" % (basedir, custom_fps)
            latticedir = datadir + os.sep + "plot_lattice"
            videopath = grid_video(datadir, vidname, imagedir=latticedir, fps=custom_fps)

    if print_grid_stats_from_file:
        filestr_data = RUNS_FOLDER + os.sep + "gen_basin_grid_C.txt"
        basin_grid_data = load_basin_grid(filestr_data)
        grid_stats(basin_grid_data)
        """
        ensemble = 960
        basin_grid_A = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_A.txt") / ensemble
        basin_grid_B = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_B.txt") / ensemble
        basin_grid_C = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_C.txt") / ensemble
        basin_grid_D = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_D.txt") / ensemble
        basin_grid_E = load_basin_grid(RUNS_FOLDER + os.sep + "gen_basin_grid_E.txt") / ensemble
        for idx, label in enumerate(celltype_labels):
            print idx, "%.2f vs %.2f" % (basin_grid_A[idx,-1], basin_grid_C[idx,-1]), label
            print idx, "%.2f vs %.2f" % (basin_grid_B[idx,-1], basin_grid_D[idx,-1]), label
        """
