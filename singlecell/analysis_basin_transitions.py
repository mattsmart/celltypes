import os
import utils.init_multiprocessing  # import before numpy
import numpy as np
import time
from multiprocessing import Pool, cpu_count, current_process

from singlecell.analysis_basin_plotting import plot_proj_timeseries, plot_basin_occupancy_timeseries, plot_basin_step
from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import ASYNC_BATCH, FIELD_APPLIED_PROTOCOL
from singlecell.singlecell_fields import field_setup
from singlecell.singlecell_simsetup import singlecell_simsetup, unpack_simsetup
from utils.file_io import run_subdir_setup, runinfo_append, RUNS_FOLDER

# analysis settings
ANALYSIS_SUBDIR = "basin_transitions"
ANNEAL_BETA = 1.3
ANNEAL_PROTOCOL = "protocol_A"
OCC_THRESHOLD = 0.7
SPURIOUS_LIST = ["mixed"]
PROFILE_PREFIX = "profile_row_"

# analysis plotting
highlights_CLPside = {6: 'k', 8: 'blue', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
highlights_simple = {6: 'k', 8: 'blue', 10: 'steelblue'}
highlights_both = {6: 'k', 8: 'blue', 10: 'steelblue', 9: 'forestgreen', 7: 'red', 16: 'deeppink', 11: 'darkorchid'}
DEFAULT_HIGHLIGHTS = highlights_CLPside


def anneal_setup(protocol=ANNEAL_PROTOCOL):
    """
    Start in basin of interest at some intermediate temperature that allows basin escape
    (e.g. beta_init = 1 / T_init = 1.3)
    For each trajectory:
    - Modify temperature once it has left basin
      (define by projection vs a strict cutoff, e.g. if projection[mem_init] < 0.6, then the trajectory is wandering)
    - Modification schedule: Decrease temp each timestep (beta += beta_step) until some ceiling is reached (beta_end)
    - Note currently "one timestep" is N spin flips.
    - If particle re-enters basin (using same cutoff as above), reset beta to beta_init and repeat procedure.
    """
    assert protocol in ["constant", "protocol_A", "protocol_B"]
    anneal_dict = {'protocol': protocol,
                   'beta_start': ANNEAL_BETA}
    if protocol == "protocol_A":
        anneal_dict.update({'beta_end': 2.0,
                            'beta_step': 0.1,
                            'wandering_threshold': 0.6})
    elif protocol == "protocol_B":
        anneal_dict.update({'beta_end': 3.0,
                            'beta_step': 0.5,
                            'wandering_threshold': 0.6})
    else:
        assert protocol == "constant"
        anneal_dict.update({'beta_end': ANNEAL_BETA,
                            'beta_step': 0.0,
                            'wandering_threshold': 0.6})
    return anneal_dict


def anneal_iterate(proj_onto_init, beta_current, step, wandering, anneal_dict, verbose=False):
    # TODO implement "acceleration" if it stays in basin or keeps re-entering basin
    if proj_onto_init < anneal_dict['wandering_threshold']:
        if verbose:
            print("+++++++++++++++++++++++++++++++++ Wandering condition passed at step %d" % step)
        wandering = True
    elif wandering:
        if verbose:
            print("+++++++++++++++++++++++++++++++++ Re-entered orig basin after wandering at step %d" % step)
        wandering = False
        beta_current = anneal_dict['beta_start']
    if wandering and beta_current < anneal_dict['beta_end']:
        beta_current = beta_current + anneal_dict['beta_step']
    return beta_current, wandering


def get_init_info(init_cond, simsetup):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    Return:
    - init state (Nx1 array) and init_id (str)
    """
    if isinstance(init_cond, np.ndarray):
        init_state = init_cond
        init_id = 'specific'
    else:
        assert isinstance(init_cond, str)
        init_state = simsetup['XI'][:, simsetup['CELLTYPE_ID'][init_cond]]
        init_id = init_cond
    return init_state, init_id


def save_and_plot_basinstats(io_dict, proj_data, occ_data, num_steps, ensemble, prefix='', simsetup=None,
                             occ_threshold=OCC_THRESHOLD, plot=True, highlights=DEFAULT_HIGHLIGHTS):
    # filename prep
    if prefix[-1] != '_':
        prefix += '_'
    # simsetup unpack for labelling plots
    if simsetup is None:
        simsetup = singlecell_simsetup()
    memory_labels = simsetup['CELLTYPE_LABELS']
    memory_id = simsetup['CELLTYPE_ID']
    N, P, gene_labels, memory_labels, gene_id, celltype_id, xi, _, a_inv, intxn_matrix, _ = unpack_simsetup(simsetup)
    # path setup
    datapath_proj = io_dict['datadir'] + os.sep + '%sproj_timeseries.txt' % prefix
    datapath_occ = io_dict['datadir'] + os.sep + '%soccupancy_timeseries.txt' % prefix
    plotpath_proj = io_dict['plotdatadir'] + os.sep + '%sproj_timeseries.png' % prefix
    plotpath_occ = io_dict['plotdatadir'] + os.sep + '%soccupancy_timeseries.png' % prefix
    plotpath_basin_endpt = io_dict['plotdatadir'] + os.sep + '%sendpt_distro.png' % prefix
    # save data to file
    np.savetxt(datapath_proj, proj_data, delimiter=',', fmt='%.4f')
    np.savetxt(datapath_occ, occ_data, delimiter=',', fmt='%i')
    # plot and save figs
    if plot:
        plot_proj_timeseries(proj_data, num_steps, ensemble, memory_labels, plotpath_proj, highlights=highlights)
        plot_basin_occupancy_timeseries(occ_data, num_steps, ensemble, memory_labels, occ_threshold, SPURIOUS_LIST, plotpath_occ, highlights=highlights)
        plot_basin_step(occ_data[:, -1], num_steps, ensemble, memory_labels, memory_id, SPURIOUS_LIST, plotpath_basin_endpt, highlights=highlights)
    return


def load_basinstats(rowdata_dir, celltype):
    proj_name = "%s_proj_timeseries.txt" % celltype
    occ_name = "%s_occupancy_timeseries.txt" % celltype
    proj_timeseries_array = np.loadtxt(rowdata_dir + os.sep + proj_name, delimiter=',', dtype=float)
    basin_occupancy_timeseries = np.loadtxt(rowdata_dir + os.sep + occ_name, delimiter=',', dtype=int)
    return proj_timeseries_array, basin_occupancy_timeseries


def fetch_from_run_info(txtpath, obj_labels):
    """
    Args:
        txtpath: path to standardized 'run_info.txt' file
        obj_labels: list of the form ['str_to_look_for', ...] type it will be read as is defined in local dict
    Returns:
        list of corresponding objects (or None's if they weren't found) in the same order
    """
    label_obj_map = {'ensemble': int,
                     'num_steps': int}
    assert all([a in list(label_obj_map.keys()) for a in obj_labels])
    linelist = [line.rstrip() for line in open(txtpath)]
    fetched_values = [None for _ in obj_labels]
    for line in linelist:
        line = line.split(',')
        for idx, label in enumerate(obj_labels):
            if label == line[0]:
                fetched_values[idx] = label_obj_map[label](line[1])
    return fetched_values


def wrapper_get_basin_stats(fn_args_dict):
    np.random.seed()
    if fn_args_dict['kwargs'] is not None:
        return get_basin_stats(*fn_args_dict['args'], **fn_args_dict['kwargs'])
    else:
        return get_basin_stats(*fn_args_dict['args'])


def get_basin_stats(init_cond, init_state, init_id, ensemble, ensemble_idx, simsetup, num_steps=100,
                    anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_APPLIED_PROTOCOL, occ_threshold=OCC_THRESHOLD,
                    async_batch=ASYNC_BATCH, verbose=False, profile=False):

    if profile:
        start_outer = time.time()  # TODO remove

    # simsetup unpack
    N, _, gene_labels, memory_labels, gene_id, celltype_id, xi, _, a_inv, intxn_matrix, _ = unpack_simsetup(simsetup)

    # prep applied field TODO: how to include applied field neatly
    field_dict = field_setup(simsetup, protocol=field_protocol)
    assert not field_dict['time_varying']  # TODO not yet supported
    app_field = field_dict['app_field']
    app_field_strength = field_dict['app_field_strength']

    transfer_dict = {}
    proj_timeseries_array = np.zeros((len(memory_labels), num_steps))
    basin_occupancy_timeseries = np.zeros((len(memory_labels) + len(SPURIOUS_LIST), num_steps), dtype=int)
    assert len(SPURIOUS_LIST) == 1
    mixed_index = len(memory_labels)  # i.e. last elem

    anneal_dict = anneal_setup(protocol=anneal_protocol)
    wandering = False

    if profile:
        start_inner = time.time()  # TODO remove

    for cell_idx in range(ensemble_idx, ensemble_idx + ensemble):
        if verbose:
            print("Simulating cell:", cell_idx)
        cell = Cell(init_state, init_id, memory_labels, gene_labels)

        beta = anneal_dict['beta_start']  # reset beta to use in each trajectory

        for step in range(num_steps):

            # report on each mem proj ranked
            projvec = cell.get_memories_projection(a_inv, xi)
            proj_timeseries_array[:, step] += projvec
            absprojvec = np.abs(projvec)
            topranked = np.argmax(absprojvec)
            if verbose:
                print("\ncell %d step %d" % (cell_idx, step))

            # print some timestep proj ranking info
            if verbose:
                for rank in range(10):
                    sortedmems_smalltobig = np.argsort(absprojvec)
                    sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
                    topranked = sortedmems_bigtosmall[0]
                    ranked_mem_idx = sortedmems_bigtosmall[rank]
                    ranked_mem = memory_labels[ranked_mem_idx]
                    print(rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx])

            if projvec[topranked] > occ_threshold:
                basin_occupancy_timeseries[topranked, step] += 1
            else:
                basin_occupancy_timeseries[mixed_index, step] += 1

            """ comment out for speedup
            if topranked != celltype_id[init_cond] and projvec[topranked] > occ_threshold:
                if cell_idx not in transfer_dict:
                    transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}
                else:
                    if topranked not in transfer_dict[cell_idx]:
                        transfer_dict[cell_idx] = {topranked: (step, projvec[topranked])}
            """
            # annealing block
            proj_onto_init = projvec[celltype_id[init_cond]]
            beta, wandering = anneal_iterate(
                proj_onto_init, beta, step, wandering, anneal_dict, verbose=verbose)

            # main call to update
            if step < num_steps:
                #cell.update_state(intxn_matrix, beta=beta, app_field=None, async_batch=async_batch)
                cell.update_state(intxn_matrix, beta=beta, field_applied=app_field,
                                  field_applied_strength=app_field_strength, async_batch=async_batch)

    if profile:
        end_inner = time.time()
        total_time = end_inner - start_outer
        if verbose:
            print("TIMINGS for %s, process %s, ensemble_start %d, last job %d" % \
                  (init_cond, current_process(), ensemble_idx, cell_idx))
            print("start outer | start inner | end --- init_time | total time")
            print("%.2f | %.2f | %.2f --- %.2f | %.2f " % \
                  (start_outer, start_inner, end_inner, start_inner - start_outer, total_time))
    else:
        total_time = None

    return transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, total_time


def fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, simsetup=None, num_steps=100,
                     occ_threshold=OCC_THRESHOLD, anneal_protocol=ANNEAL_PROTOCOL, field_protocol=FIELD_APPLIED_PROTOCOL,
                     async_batch=ASYNC_BATCH, verbose=False, profile=False):
    # simsetup unpack
    if simsetup is None:
        simsetup = singlecell_simsetup()
    # prepare fn args and kwargs for wrapper
    kwargs_dict = {'num_steps': num_steps, 'anneal_protocol': anneal_protocol, 'field_protocol': field_protocol,
                   'occ_threshold': occ_threshold, 'async_batch': async_batch, 'verbose': verbose, 'profile': profile}
    fn_args_dict = [0]*num_processes
    if verbose:
        print("NUM_PROCESSES:", num_processes)
    assert ensemble % num_processes == 0
    for i in range(num_processes):
        subensemble = ensemble / num_processes
        cell_startidx = i * subensemble
        if verbose:
            print("process:", i, "job size:", subensemble, "runs")
        fn_args_dict[i] = {'args': (init_cond, init_state, init_id, subensemble, cell_startidx, simsetup),
                           'kwargs': kwargs_dict}
    # generate results list over workers
    t0 = time.time()
    pool = Pool(num_processes)
    print("pooling")
    results = pool.map(wrapper_get_basin_stats, fn_args_dict)
    print("done")
    pool.close()
    pool.join()
    if verbose:
        print("TIMER:", time.time() - t0)
    # collect pooled results
    summed_transfer_dict = {}  # TODO remove?
    summed_proj_timeseries_array = np.zeros((len(simsetup['CELLTYPE_LABELS']), num_steps))
    summed_basin_occupancy_timeseries = np.zeros((len(simsetup['CELLTYPE_LABELS']) + 1, num_steps), dtype=int)  # could have some spurious here too? not just last as mixed
    if profile:
        worker_times = np.zeros(num_processes)
    else:
        worker_times = None
    for i, result in enumerate(results):
        transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, worker_time = result
        summed_transfer_dict.update(transfer_dict)  # TODO check
        summed_proj_timeseries_array += proj_timeseries_array
        summed_basin_occupancy_timeseries += basin_occupancy_timeseries
        if profile:
            worker_times[i] = worker_time
    #check2 = np.sum(summed_basin_occupancy_timeseries, axis=0)

    # notmalize proj timeseries
    summed_proj_timeseries_array = summed_proj_timeseries_array / ensemble  # want ensemble average

    return summed_transfer_dict, summed_proj_timeseries_array, summed_basin_occupancy_timeseries, worker_times


def ensemble_projection_timeseries(init_cond, ensemble, num_processes, simsetup=None, num_steps=100,
                                   occ_threshold=OCC_THRESHOLD, anneal_protocol=ANNEAL_PROTOCOL,
                                   field_protocol=FIELD_APPLIED_PROTOCOL, async_batch=ASYNC_BATCH, output=True, plot=True,
                                   profile=False):
    """
    Args:
    - init_cond: np array of init state OR string memory label
    - ensemble: ensemble of particles beginning at init_cond
    - num_steps: how many steps to iterate (each step updates every spin once)
    - occ_threshold: projection value cutoff to say state is in a basin (default: 0.7)
    - anneal_protocol: define how temperature changes during simulation
    What:
    - Track a timeseries of: ensemble mean projection onto each memory
    - Optionally plot
    Eeturn:
    - timeseries of projections onto store memories (dim p x T)
    """

    # simsetup unpack
    if simsetup is None:
        simsetup = singlecell_simsetup()

    # prep io
    if output:
        io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
    else:
        assert not plot
        io_dict = None

    # profiler setup
    profile_path = RUNS_FOLDER + os.sep + PROFILE_PREFIX + "%dens_%dsteps.txt" % (ensemble, num_steps)
    if profile:
        time_start = time.time()

    # generate initial state
    init_state, init_id = get_init_info(init_cond, simsetup)

    # simulate ensemble - pooled wrapper call
    transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, worker_times = \
        fast_basin_stats(init_cond, init_state, init_id, ensemble, num_processes, simsetup=simsetup, num_steps=num_steps,
                         anneal_protocol=anneal_protocol, field_protocol=field_protocol, occ_threshold=occ_threshold,
                         async_batch=async_batch, verbose=False, profile=profile)

    # save data and plot figures
    if output:
        save_and_plot_basinstats(io_dict, proj_timeseries_array, basin_occupancy_timeseries, num_steps, ensemble,
                                 simsetup=simsetup, prefix=init_id, occ_threshold=occ_threshold, plot=plot)
    """
    # print transfer dict
    for idx in xrange(ensemble):
        if idx in transfer_dict:
            print idx, transfer_dict[idx], [simsetup['CELLTYPE_LABELS'][a] for a in transfer_dict[idx].keys()]
    """
    if profile:
        time_end = time.time()
        time_total = time_end - time_start
        with open(profile_path, 'a+') as f:
            f.write('%d,%.2f,%.2f,%.2f\n' % (num_processes, time_total, min(worker_times), max(worker_times)))
            #f.write(','.join(str(s) for s in [num_processes, time_total]) + '\n')

    return proj_timeseries_array, basin_occupancy_timeseries, worker_times, io_dict


def basin_transitions(init_cond, ensemble, num_steps, beta, simsetup):
    # TODO note analysis basin grid fulfills this functionality, not great spurious handling though
    """
    Track jumps from basin 'i' to basin 'j' for all 'i'

    Defaults:
    - temperature: default is intermediate (1/BETA from singlecell_constants)
    - ensemble: 10,000 cells start in basin 'i'
    - time: fixed, 100 steps (option for unspecified; stop when ensemble dissipates)

    Output:
    - matrix of basin transition probabilities (i.e. discrete time markov chain)

    Spurious basin notes:
    - unclear how to identify spurious states dynamically
    - suppose
    - define new spurious state if, within some window of time T:
        - (A) the state does not project on planned memories within some tolerance; and
        - (B) the state has some self-similarity over time
    - if a potential function is known (e.g. energy H(state)) then a spurious state
      could be formally defined as a minimizer; however this may be numerically expensive to check
    """
    """
    app_field = construct_app_field_from_genes(IPSC_CORE_GENES, num_steps)
    proj_timeseries_array = np.zeros((num_steps, P))
    """

    # add 1 as spurious sink dimension? this treats spurious as global sink state
    basins_dim = len(simsetup['CELLTYPE_LABELS']) + 1
    spurious_index = len(simsetup['CELLTYPE_LABELS'])
    transition_data = np.zeros((basins_dim, basins_dim))

    for idx, memory_label in enumerate(simsetup['CELLTYPE_LABELS']):
        # TODO
        print(idx, memory_label)
        """
        cellstate_array, current_run_folder, data_folder, plot_lattice_folder, plot_data_folder = singlecell_sim(init_id=memory_label, iterations=num_steps, app_field=app_field, app_field_strength=10.0,
                                                                                                                 flag_burst_error=FLAG_BURST_ERRORS, flag_write=False, analysis_subdir=analysis_subdir,
                                                                                                                 plot_period=num_steps*2)
        proj_timeseries_array[:, idx] = get_memory_proj_timeseries(cellstate_array, esc_idx)[:]
        """
        # TODO: transiton_data_row = ...
        transiton_data_row = 0

        transition_data[idx, :] = transiton_data_row


    # cleanup output folders from main()
    # TODO...

    # save transition array and run info to file
    # TODO...

    # plot output
    # TODO...

    return transition_data


if __name__ == '__main__':
    gen_basin_data = False
    plot_grouped_data = False
    profile = False
    plot_groups_of_transitions = True

    # prep simulation globals
    simsetup = singlecell_simsetup()

    if gen_basin_data:
        # common: 'HSC' / 'Common Lymphoid Progenitor (CLP)' / 'Common Myeloid Progenitor (CMP)' /
        #         'Megakaryocyte-Erythroid Progenitor (MEP)' / 'Granulocyte-Monocyte Progenitor (GMP)' / 'thymocyte DN'
        #         'thymocyte - DP' / 'neutrophils' / 'monocytes - classical'
        init_cond = 'macrophage'  # note HSC index is 6 in mehta mems
        ensemble = 128
        num_steps = 100
        num_proc = cpu_count() / 2  # seems best to use only physical core count (1 core ~ 3x slower than 4)
        anneal_protocol = "protocol_A"
        field_protocol = "miR_21"  # "yamanaka" or "miR_21" or None
        async_batch = True
        plot = True
        parallel = True

        # run and time basin ensemble sim
        t0 = time.time()
        if parallel:
            proj_timeseries_array, basin_occupancy_timeseries, worker_times, io_dict = \
                ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps, simsetup=simsetup,
                                               occ_threshold=OCC_THRESHOLD, anneal_protocol=anneal_protocol,
                                               field_protocol=field_protocol, async_batch=async_batch, plot=plot)
        else:
            # Unparallelized for testing/profiling:
            init_state, init_id = get_init_info(init_cond, simsetup)
            io_dict = run_subdir_setup(run_subfolder=ANALYSIS_SUBDIR)
            transfer_dict, proj_timeseries_array, basin_occupancy_timeseries, worker_times = \
                get_basin_stats(init_cond, init_state, init_id, ensemble, 0, simsetup, num_steps=num_steps,
                                anneal_protocol=anneal_protocol, field_protocol=field_protocol,
                                occ_threshold=OCC_THRESHOLD, async_batch=async_batch, verbose=False, profile=True)
            proj_timeseries_array = proj_timeseries_array / ensemble  # ensure normalized (get basin stats won't do this)
        t1 = time.time() - t0
        print("Runtime:", t1)

        # append info to run info file  TODO maybe move this INTO the function?
        info_list = [['fncall', 'ensemble_projection_timeseries()'], ['init_cond', init_cond], ['ensemble', ensemble],
                     ['num_steps', num_steps], ['num_proc', num_proc], ['anneal_protocol', anneal_protocol],
                     ['occ_threshold', OCC_THRESHOLD], ['field_protocol', field_protocol],
                     ['async_batch', async_batch], ['time', t1], ['time_workers', worker_times]]
        runinfo_append(io_dict, info_list, multi=True)

    # direct data plotting
    if plot_grouped_data:
        group_dir = RUNS_FOLDER
        bases = ["output_335260","output_335261","output_335262","output_335264","output_335265"]
        types = ["HSC","HSC","mef","mef","mef"]
        labels = ["yam_1e5", "yam_0", "None", "yam_idk", "yam_1e5"]
        ensemble = 10000
        for i in range(len(bases)):
            celltypes = simsetup['CELLTYPE_LABELS']
            outdir = group_dir + os.sep + bases[i]
            outproj = outdir + os.sep + 'proj_timeseries_%s.png' % labels[i]
            outocc = outdir + os.sep + 'occ_timeseries_%s.png' % labels[i]
            outend = outdir + os.sep + 'occ_endpt_%s.png' % labels[i]
            # load and parse
            proj_data, occ_data = load_basinstats(outdir + os.sep + 'data', types[i])
            ensemble, num_steps = fetch_from_run_info(outdir + os.sep + 'run_info.txt', ['ensemble', 'num_steps'])
            assert num_steps == proj_data.shape[1]
            # plot
            plot_proj_timeseries(proj_data, num_steps, ensemble, celltypes, outproj, highlights=highlights_CLPside)
            plot_basin_occupancy_timeseries(occ_data, num_steps, ensemble, celltypes, OCC_THRESHOLD, SPURIOUS_LIST,
                                            outocc, highlights=highlights_CLPside)
            plot_basin_step(occ_data[:, -1], num_steps, ensemble, celltypes, simsetup['CELLTYPE_ID'],
                            SPURIOUS_LIST, outend, highlights=highlights_CLPside)

    if profile:
        # common: 'HSC' / 'Common Lymphoid Progenitor (CLP)' / 'Common Myeloid Progenitor (CMP)' /
        #         'Megakaryocyte-Erythroid Progenitor (MEP)' / 'Granulocyte-Monocyte Progenitor (GMP)' / 'thymocyte DN'
        #         'thymocyte - DP' / 'neutrophils' / 'monocytes - classical'
        init_cond = 'HSC'  # note HSC index is 6 in mehta mems
        num_steps = 100
        anneal_protocol = "protocol_A"
        field_protocol = None
        async_batch = ASYNC_BATCH
        plot = False
        ens_scaled = False
        if ens_scaled:
            ens_base = 16                                             # NETWORK_METHOD: all workers will do this many traj
            proc_lists = {p: list(range(1,p+1)) for p in [4,8,80]}
        else:
            ens_base = 128                                            # NETWORK_METHOD: divide this number amongst all workers
            proc_lists = {4: [1,2,3,4],
                          8: [1,2,4,8], #[1,2,3,4,5,6,8],
                          64: [1,2,4,8,16,32,64],
                          80: [1,2,3,4,5,6,8,10,12,15,16,20,24,30,40,48,60,80]}

        # run and time basin ensemble sim
        for num_proc in proc_lists[cpu_count()]:
            if ens_scaled:
                ensemble = ens_base * num_proc
            else:
                ensemble = ens_base
            print("Start timer for num_proc %d (%d ens x %d steps)" % (num_proc, ensemble, num_steps))
            t0 = time.time()
            proj_timeseries_array, basin_occupancy_timeseries, worker_times, io_dict = \
                ensemble_projection_timeseries(init_cond, ensemble, num_proc, num_steps=num_steps, simsetup=simsetup,
                                               occ_threshold=OCC_THRESHOLD, anneal_protocol=anneal_protocol,
                                               field_protocol=field_protocol, async_batch=async_batch, plot=plot,
                                               output=True, profile=True)
            t1 = time.time() - t0
            print("Runtime:", t1)

            # append info to run info file  TODO maybe move this INTO the function?
            info_list = [['fncall', 'ensemble_projection_timeseries()'], ['init_cond', init_cond],
                         ['ensemble', ensemble],
                         ['num_steps', num_steps], ['num_proc', num_proc], ['anneal_protocol', anneal_protocol],
                         ['occ_threshold', OCC_THRESHOLD], ['field_protocol', field_protocol],
                         ['async_batch', async_batch], ['time', t1], ['time_workers', worker_times]]
            runinfo_append(io_dict, info_list, multi=True)

    # plot specific directory data from basin transitions run
    if plot_groups_of_transitions:
        groupdir = RUNS_FOLDER + os.sep + 'single_celltype_transitions'
        basedirs = ['Macrophage (A)', 'Macrophage (B)', 'Macrophage (C)', 'Macrophage (D)']  # celltype labels
        subdirs = ['noField', 'mir21_1', 'mir21_2', 'mir21_3']  # field labels
        for basedir in basedirs:
            for subdir in subdirs:
                datadir = groupdir + os.sep + basedir + os.sep + subdir + os.sep + 'data'
                print("working in", datadir)
                # load proj data and occ data
                proj_data, occ_data = load_basinstats(datadir, basedir)
                ens = float(np.sum(occ_data[:, 0]))
                print(proj_data.shape)
                # setup timepoints
                total_steps = proj_data.shape[1]
                num_timepoints = 0
                timepoints = [a*int(total_steps/num_timepoints) for a in range(num_timepoints)]
                timepoints.append(total_steps-1)
                for step in timepoints:
                    # sort proj and occ data at each timepoint
                    projvec = proj_data[:, step]
                    absprojvec = np.abs(projvec)
                    occvec = occ_data[:, step]  # TODO
                    # print some timestep proj ranking info
                    print("\nRanking transitions (by proj) from %s, %s at step %d" % (basedir, subdir, step))
                    sortedmems_smalltobig = np.argsort(absprojvec)
                    sortedmems_bigtosmall = sortedmems_smalltobig[::-1]
                    for rank in range(10):
                        ranked_mem_idx = sortedmems_bigtosmall[rank]
                        ranked_mem = simsetup['CELLTYPE_LABELS'][ranked_mem_idx]
                        print(rank, ranked_mem_idx, ranked_mem, projvec[ranked_mem_idx], absprojvec[ranked_mem_idx])
                    # print some timestep occ ranking info
                    print("\nRanking transitions (by occ) from %s, %s at step %d" % (basedir, subdir, step))
                    occ_labels = simsetup['CELLTYPE_LABELS'] + SPURIOUS_LIST
                    sortedmems_smalltobig = np.argsort(occvec)
                    sortedmems_bigtosmall_occ = sortedmems_smalltobig[::-1]
                    for rank in range(10):
                        ranked_mem_idx = sortedmems_bigtosmall_occ[rank]
                        ranked_mem = occ_labels[ranked_mem_idx]
                        print(rank, ranked_mem_idx, ranked_mem, occvec[ranked_mem_idx], occvec[ranked_mem_idx] / ens)
                    # plot sorted data with labels
                    outpath = groupdir + os.sep + 'occ_%s_%s_step_%d.png' % (basedir, subdir, step)
                    sorted_occ = [occ_data[idx, step] for idx in sortedmems_bigtosmall_occ]
                    sorted_labels = [occ_labels[idx] for idx in sortedmems_bigtosmall_occ]
                    print('\n', len(sorted_occ))
                    plot_basin_step(sorted_occ, step, ens, sorted_labels, simsetup['CELLTYPE_ID'], [], outpath,
                                    highlights=None, autoscale=True, inset=True, title_add='(%s)' % subdir, init_mem=basedir)
