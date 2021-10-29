import numpy as np
import os
import random
import matplotlib.pyplot as plt

from multicell.graph_adjacency import \
    lattice_square_int_to_loc
from multicell.multicell_constants import \
    GRIDSIZE, SEARCH_RADIUS_CELL, NUM_LATTICE_STEPS, VALID_BUILDSTRINGS, VALID_EXOSOME_STRINGS, \
    BUILDSTRING, EXOSTRING, LATTICE_PLOT_PERIOD, MEANFIELD, EXOSOME_REMOVE_RATIO, \
    BLOCK_UPDATE_LATTICE, AUTOCRINE
from multicell.multicell_lattice import \
    build_lattice_main, get_cell_locations, prep_lattice_data_dict, write_state_all_cells, \
    write_grid_state_int
from multicell.multicell_metrics import \
    calc_lattice_energy, calc_compression_ratio, get_state_of_lattice
from multicell.multicell_visualize_old import \
    lattice_uniplotter, reference_overlap_plotter, lattice_projection_composite
from singlecell.singlecell_constants import FIELD_SIGNAL_STRENGTH, FIELD_APPLIED_STRENGTH, BETA
from singlecell.singlecell_fields import construct_app_field_from_genes
from singlecell.singlecell_simsetup import singlecell_simsetup
from utils.file_io import run_subdir_setup, runinfo_append, write_general_arr, read_general_arr


def run_mc_sim(lattice, num_lattice_steps, data_dict, io_dict, simsetup, exosome_string=EXOSTRING,
               exosome_remove_ratio=0.0, ext_field_strength=FIELD_SIGNAL_STRENGTH, app_field=None,
               app_field_strength=FIELD_APPLIED_STRENGTH, beta=BETA, plot_period=LATTICE_PLOT_PERIOD,
               flag_uniplots=False, state_int=False, meanfield=MEANFIELD):
    """
    Form of data_dict:
        {'memory_proj_arr':
            {memory_idx: np array [N x num_steps] of projection each grid cell onto memory idx}
         'grid_state_int': n x n x num_steps of int at each site
           (int is inverse of binary string from state)
    Notes:
        -can replace update_with_signal_field with update_state to simulate ensemble
        of non-interacting n**2 cells
    """
    def input_checks(app_field):
        n = len(lattice)
        assert n == len(lattice[0])  # work with square lattice for simplicity
        num_cells = n * n
        assert SEARCH_RADIUS_CELL < n / 2.0  # to prevent signal double counting

        if app_field is not None:
            if len(app_field.shape) > 1:
                assert app_field.shape[0] == simsetup['N']
                assert len(app_field[1]) == num_lattice_steps
            else:
                app_field = np.array([app_field for _ in range(num_lattice_steps)]).T
            app_field_step = app_field[:, 0]
        else:
            app_field_step = None
        return n, num_cells, app_field, app_field_step

    def update_datadict_timestep_cell(lattice, loc, memory_idx_list, timestep_idx):
        cell = lattice[loc[0]][loc[1]]
        # store the projections
        proj = cell.get_memories_projection(simsetup['A_INV'], simsetup['XI'])
        for mem_idx in memory_idx_list:
            data_dict['memory_proj_arr'][mem_idx][loc_to_idx[loc], timestep_idx] = proj[mem_idx]
        # store the integer representation of the state
        if state_int:
            data_dict['grid_state_int'][loc[0], loc[1], timestep_idx] = cell.get_current_label()
        return proj

    def update_datadict_timestep_global(lattice, timestep_idx):
        data_dict['lattice_energy'][timestep_idx, :] = calc_lattice_energy(
            lattice, simsetup, app_field_step, app_field_strength, ext_field_strength, SEARCH_RADIUS_CELL,
            exosome_remove_ratio, exosome_string, meanfield)
        data_dict['compressibility_full'][timestep_idx, :] = calc_compression_ratio(
            get_state_of_lattice(lattice, simsetup, datatype='full'),
            eta_0=None, datatype='full', elemtype=np.int, method='manual')

    def lattice_plot_init(lattice, memory_idx_list):
        lattice_projection_composite(lattice, 0, n, io_dict['latticedir'], simsetup, state_int=state_int)
        reference_overlap_plotter(lattice, 0, n, io_dict['latticedir'], simsetup, state_int=state_int)
        if flag_uniplots:
            for mem_idx in memory_idx_list:
                lattice_uniplotter(lattice, 0, n, io_dict['latticedir'], mem_idx, simsetup)

    def meanfield_global_field():
        # TODO careful: not clear best way to update exo field as cell state changes in a time step,
        #  refactor exo fn?
        assert exosome_string == 'no_exo_field'
        print('Initializing mean field...')
        # TODO decide if want scale factor to be rescaled by total popsize (i.e. *mean*field or total field?)
        state_total = np.zeros(simsetup['N'])
        field_global = np.zeros(simsetup['N'])
        # TODO ok that cell is neighbour with self as well? should remove diag
        neighbours = [[a, b] for a in range(len(lattice[0])) for b in range(len(lattice))]
        if simsetup['FIELD_SEND'] is not None:
            for loc in neighbours:
                state_total += lattice[loc[0]][loc[1]].get_current_state()
            state_total_01 = (state_total + num_cells) / 2
            field_paracrine = np.dot(simsetup['FIELD_SEND'], state_total_01)
            field_global += field_paracrine
        if exosome_string != 'no_exo_field':
            field_exo, _ = lattice[0][0].\
                get_local_exosome_field(lattice, None, None, exosome_string=exosome_string,
                                        exosome_remove_ratio=exosome_remove_ratio, neighbours=neighbours)
            field_global += field_exo
        return field_global

    def parallel_block_update_lattice(J_block, s_block_current, applied_field_block, total_spins):
        # TODO (determ vs 0 temp?)
        total_field = np.zeros(total_spins)
        internal_field = np.dot(J_block, s_block_current)
        total_field += internal_field

        # TODO deal with time dependent applied field on the whole lattice here (e.g. if using
        #  housekeeping or other)
        if applied_field_block is not None:
            applied_field_block_scaled = app_field_strength * applied_field_block
            total_field += applied_field_block_scaled

        # probability that site i will be "up" after the timestep
        prob_on_after_timestep = 1 / (1 + np.exp(-2 * beta * total_field))
        rsamples = np.random.rand(total_spins)
        for idx in range(total_spins):
            if prob_on_after_timestep[idx] > rsamples[idx]:
                s_block_current[idx] = 1.0
            else:
                s_block_current[idx] = -1.0
        return s_block_current

    def build_block_matrices_from_search_radius(n, num_cells, search_radius, gamma,
                                                aotocrine=AUTOCRINE, plot_adjacency=False):
        W_scaled = gamma * simsetup['FIELD_SEND']

        # Term A: self interactions for each cell (diagonal blocks of multicell J_block)
        if aotocrine:
            J_diag_blocks = np.kron(np.eye(num_cells), simsetup['J'] + W_scaled)
        else:
            J_diag_blocks = np.kron(np.eye(num_cells), simsetup['J'])

        # Term B:
        adjacency_arr_uptri = np.zeros((num_cells, num_cells))
        # build only upper diagonal part of A
        for a in range(num_cells):
            grid_loc_a = lattice_square_int_to_loc(a, n)  # map cell a & b index to grid loc (i, j)
            arow, acol = grid_loc_a[0], grid_loc_a[1]
            arow_low = arow - search_radius
            arow_high = arow + search_radius
            acol_low = acol - search_radius
            acol_high = acol + search_radius
            for b in range(a+1, num_cells):
                grid_loc_b = lattice_square_int_to_loc(b, n)  # map cell a & b index to grid loc (i, j)
                # is neighbor?
                if (arow_low <= grid_loc_b[0] <= arow_high) and (acol_low <= grid_loc_b[1] <= acol_high):
                    adjacency_arr_uptri[a, b] = 1
        adjacency_arr_lowtri = adjacency_arr_uptri.T
        adjacency_arr = adjacency_arr_lowtri + adjacency_arr_uptri

        # Term 2 of J_multicell (cell-cell interactions)
        J_offdiag_blocks = np.kron(adjacency_arr_lowtri, W_scaled.T) \
                           + np.kron(adjacency_arr_uptri, W_scaled)

        # build final J multicell matrix
        J_block = J_diag_blocks + gamma * J_offdiag_blocks

        if plot_adjacency:
            plt.imshow(adjacency_arr)
            plt.show()

        return J_block, adjacency_arr

    def build_block_state_from_lattice(lattice, n, num_cells, simsetup):
        N = simsetup['N']
        total_spins = num_cells * N
        s_block = np.zeros(total_spins)
        for a in range(num_cells):
            arow, acol = lattice_square_int_to_loc(a, n)
            cellstate = np.copy(
                lattice[arow][acol].get_current_state())
            s_block[a * N: (a+1) * N] = cellstate
        return s_block

    def update_lattice_using_state_block(lattice, n, num_cells, simsetup, s_block):
        N = simsetup['N']
        total_spins = num_cells * N
        for a in range(num_cells):
            arow, acol = lattice_square_int_to_loc(a, n)
            cell = lattice[arow][acol]
            cellstate = np.copy(s_block[a * N: (a + 1) * N])
            # update cell state specifically
            lattice[arow][acol].state = cellstate
            # update whole cell state array (append new state for the current timepoint)
            state_array_ext = np.zeros((N, np.shape(cell.state_array)[1] + 1))
            state_array_ext[:, :-1] = cell.state_array  # TODO: make sure don't need array copy
            state_array_ext[:,-1] = cellstate
            cell.state_array = state_array_ext
            # update steps attribute
            cell.steps += 1
        return lattice

    # input processing
    n, num_cells, app_field, app_field_step = input_checks(app_field)
    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}
    memory_idx_list = list(data_dict['memory_proj_arr'].keys())

    # assess & plot initial state
    for loc in cell_locations:
        update_datadict_timestep_cell(lattice, loc, memory_idx_list, 0)
    update_datadict_timestep_global(lattice, 0)  # measure initial state
    lattice_plot_init(lattice, memory_idx_list)  # plot initial state

    # special update method for meanfield case (infinite search radius)
    if meanfield:
        state_total, field_global = meanfield_global_field()

    if BLOCK_UPDATE_LATTICE:
        assert not meanfield  # TODO how does this flag interact with meanfield flag?
        # Psuedo 1: build J = I dot J0 + A dot W
        # I is M x M, A determined by the type of graph (could explore other, non-lattice, types)
        total_spins = num_cells * simsetup['N']
        J_block, adjacency_arr = build_block_matrices_from_search_radius(
            n, num_cells, SEARCH_RADIUS_CELL, ext_field_strength, aotocrine=AUTOCRINE)

        # Pseudo 2: store lattice state as blocked vector s_hat
        state_block = build_block_state_from_lattice(lattice, n, num_cells, simsetup)

        # Pseudo 3: applied_field_block timeseries or None
        # TODO

    for turn in range(1, num_lattice_steps):
        print('Turn ', turn)

        if BLOCK_UPDATE_LATTICE:
            # TODO applied field block
            # block update rule for the lattice (represented by state_block)
            state_block = parallel_block_update_lattice(J_block, state_block, None, total_spins)

            # TODO applied field block
            # TODO better usage of the lattice object, this refilling is inefficient
            #  especially the state array part
            # fill lattice object based on updated state_block
            lattice = update_lattice_using_state_block(lattice, n, num_cells, simsetup, state_block)

        else:
            random.shuffle(cell_locations)
            for idx, loc in enumerate(cell_locations):
                cell = lattice[loc[0]][loc[1]]
                if app_field is not None:
                    app_field_step = app_field[:, turn]
                if meanfield:
                    cellstate_pre = np.copy(cell.get_current_state())
                    cell.update_with_meanfield(
                        simsetup['J'], field_global, beta=beta, app_field=app_field_step,
                        field_signal_strength=ext_field_strength, field_app_strength=app_field_strength)
                    # TODO update field_avg based on new state TODO test
                    state_total += (cell.get_current_state() - cellstate_pre)
                    state_total_01 = (state_total + num_cells) / 2
                    field_global = np.dot(simsetup['FIELD_SEND'], state_total_01)
                    print(field_global)
                    print(state_total)
                else:
                    cell.update_with_signal_field(
                        lattice, SEARCH_RADIUS_CELL, n, simsetup['J'], simsetup, beta=beta,
                        exosome_string=exosome_string, exosome_remove_ratio=exosome_remove_ratio,
                        field_signal_strength=ext_field_strength, field_app=app_field_step,
                        field_app_strength=app_field_strength)

                # update cell specific datdict entries for the current timestep
                cell_proj = update_datadict_timestep_cell(lattice, loc, memory_idx_list, turn)

                if turn % (120*plot_period) == 0:  # proj vis of each cell (slow; every k steps)
                    fig, ax, proj = cell.\
                        plot_projection(simsetup['A_INV'], simsetup['XI'], proj=cell_proj,
                                        use_radar=False, pltdir=io_dict['latticedir'])

        # compute lattice properties (assess global state)
        # TODO 1 - consider lattice energy at each cell update (not lattice update)
        # TODO 2 - speedup lattice energy calc by using info from state update calls...
        update_datadict_timestep_global(lattice, turn)

        if turn % plot_period == 0:  # plot the lattice
            lattice_projection_composite(
                lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            reference_overlap_plotter(
                lattice, turn, n, io_dict['latticedir'], simsetup, state_int=state_int)
            #if flag_uniplots:
            #    for mem_idx in memory_idx_list:
            #        lattice_uniplotter(lattice, turn, n, io_dict['latticedir'], mem_idx, simsetup)

    return lattice, data_dict, io_dict


def mc_sim_wrapper(simsetup, gridsize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, buildstring=BUILDSTRING,
                   field_signal_strength=FIELD_SIGNAL_STRENGTH,
                   exosome_string=EXOSTRING, exosome_remove_ratio=EXOSOME_REMOVE_RATIO,
                   field_applied=None, field_applied_strength=FIELD_APPLIED_STRENGTH,
                   flag_housekeeping=False, field_housekeeping_strength=0.0,
                   beta=BETA, meanfield=MEANFIELD,
                   plot_period=LATTICE_PLOT_PERIOD, state_int=False):
    """
    :param simsetup:               simsetup with internal and external gene regulatory rules
    :param gridsize:               (int) edge length of square multicell grid
    :param num_steps:              (int) number of many lattice timesteps
    :param buildstring:            (str) specifies init cond style of the lattice
    :param exosome_string:         (str) see valid exosome strings; adds exosomes to field_signal
    :param exosome_remove_ratio:   (float) if exosomes act, how much of the cell state to subsample?
    :param field_signal_strength:  (float) the cell-cell signalling field strength
    :param field_applied:          (None or array) the external/manual applied field
    :param field_applied_strength: (float) the external/manual applied field strength
    :param flag_housekeeping:      (bool) is there a housekeeping component to the manual field?
    :param field_housekeeping_strength: (float)
    :param beta:                   (float) inverse temperature
    :param plot_period:            (int) lattice plot period
    :param state_int:              (bool) track and plot the int rep of cell state (low N only)
    :param meanfield:              ...
    :return:
        (lattice, data_dict, io_dict)
    """

    # check args
    assert type(gridsize) is int
    assert type(num_steps) is int
    assert type(plot_period) is int
    assert buildstring in VALID_BUILDSTRINGS
    assert exosome_string in VALID_EXOSOME_STRINGS
    assert 0.0 <= exosome_remove_ratio < 1.0
    assert 0.0 <= field_signal_strength < 100.0

    # setup io dict
    io_dict = run_subdir_setup(run_subfolder='multicell_sim')
    if meanfield:
        search_radius_txt = 'None'
    else:
        search_radius_txt = SEARCH_RADIUS_CELL
    info_list = [['memories_path', simsetup['memories_path']],
                 ['script', 'multicell_simulate_old.py'],
                 ['gridsize', gridsize],
                 ['num_steps', num_steps],
                 ['buildstring', buildstring],
                 ['exosome_string', exosome_string],
                 ['exosome_remove_ratio', exosome_remove_ratio],
                 ['field_signal_strength', field_signal_strength],
                 ['field_applied_strength', field_applied_strength],
                 ['field_applied', field_applied],
                 ['flag_housekeeping', flag_housekeeping],
                 ['field_housekeeping_strength', field_housekeeping_strength],
                 ['beta', beta],
                 ['search_radius', search_radius_txt],
                 ['random_mem', simsetup['random_mem']],
                 ['random_W', simsetup['random_W']],
                 ['meanfield', meanfield],
                 ['housekeeping', simsetup['K']],
                 ['dynamics_parallel', BLOCK_UPDATE_LATTICE],
                 ['autocrine', AUTOCRINE]
                 ]
    runinfo_append(io_dict, info_list, multi=True)
    # conditionally store random mem and W
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt',
               simsetup['XI'], delimiter=',', fmt='%d')
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt',
               simsetup['FIELD_SEND'], delimiter=',', fmt='%.4f')

    # setup lattice IC
    flag_uniplots = False
    if buildstring == "mono":
        type_1_idx = 0
        list_of_type_idx = [type_1_idx]
    if buildstring == "dual":
        type_1_idx = 0
        type_2_idx = 1
        list_of_type_idx = [type_1_idx, type_2_idx]
    if buildstring == "memory_sequence":
        flag_uniplots = False
        list_of_type_idx = list(range(simsetup['P']))
        #random.shuffle(list_of_type_idx)  # TODO shuffle or not?
    if buildstring == "random":
        flag_uniplots = False
        list_of_type_idx = list(range(simsetup['P']))
    lattice = build_lattice_main(gridsize, list_of_type_idx, buildstring, simsetup)
    #print list_of_type_idx

    # prep data dictionary
    data_dict = {}
    data_dict = prep_lattice_data_dict(gridsize, num_steps, list_of_type_idx, buildstring, data_dict)
    if state_int:
        data_dict['grid_state_int'] = np.zeros((gridsize, gridsize, num_steps), dtype=int)

    # run the simulation
    lattice, data_dict, io_dict = run_mc_sim(
        lattice, num_steps, data_dict, io_dict, simsetup, exosome_string=exosome_string,
        exosome_remove_ratio=exosome_remove_ratio, ext_field_strength=field_signal_strength,
        app_field=field_applied, app_field_strength=field_applied_strength, beta=beta, state_int=state_int,
        plot_period=plot_period, flag_uniplots=flag_uniplots, meanfield=meanfield)

    # check the data dict
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print(data_dict['memory_proj_arr'][memory_idx])
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.ylabel('Projection of all cells onto type: %s' % simsetup['CELLTYPE_LABELS'][memory_idx])
        plt.xlabel('Time (full lattice steps)')
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, memory_idx,
                     exosome_remove_ratio, field_signal_strength))
        plt.clf()  #plt.show()

    # write and plot cell state timeseries
    # TODO convert to 'write data dict' and 'plot data dict' fn calls
    #write_state_all_cells(lattice, io_dict['datadir'])
    if state_int:
        write_grid_state_int(data_dict['grid_state_int'], io_dict['datadir'])
    if 'lattice_energy' in list(data_dict.keys()):
        write_general_arr(data_dict['lattice_energy'], io_dict['datadir'],
                          'lattice_energy', txt=True, compress=False)
        plt.plot(data_dict['lattice_energy'][:, 0], '--ok', label=r'$H_{\mathrm{total}}$')
        plt.plot(data_dict['lattice_energy'][:, 1], '--b', alpha=0.7, label=r'$H_{\mathrm{self}}$')
        plt.plot(data_dict['lattice_energy'][:, 2], '--g', alpha=0.7, label=r'$H_{\mathrm{app}}$')
        plt.plot(data_dict['lattice_energy'][:, 3], '--r', alpha=0.7, label=r'$H_{\mathrm{pairwise}}$')
        plt.plot(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2], '--o',
                 color='gray', label=r'$H_{\mathrm{total}} - H_{\mathrm{app}}$')
        plt.title(r'Multicell hamiltonian over time')
        plt.ylabel(r'Lattice energy')
        plt.xlabel(r'$t$ (lattice steps)')
        plt.legend()
        plt.savefig(
            io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_hamiltonian_remove%.2f_exo%.2f.png' %
            (exosome_string, buildstring, gridsize, num_steps, exosome_remove_ratio, field_signal_strength))
        # zoom on relevant part
        ylow = min(np.min(data_dict['lattice_energy'][:, [1,3]]),
                   np.min(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2]))
        yhigh = max(np.max(data_dict['lattice_energy'][:, [1,3]]),
                    np.max(data_dict['lattice_energy'][:, 0] - data_dict['lattice_energy'][:, 2]))
        plt.ylim(ylow - 0.1, yhigh + 0.1)
        plt.savefig(
            io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_hamiltonianZoom_remove%.2f_exo%.2f.png' %
            (exosome_string, buildstring, gridsize, num_steps, exosome_remove_ratio, field_signal_strength))
        plt.clf()  # plt.show()
    if 'compressibility_full' in list(data_dict.keys()):
        write_general_arr(data_dict['compressibility_full'], io_dict['datadir'],
                          'compressibility_full', txt=True, compress=False)
        plt.plot(data_dict['compressibility_full'][:,0], '--o', color='orange')
        plt.title(r'File compressibility ratio of the full lattice spin state')
        plt.ylabel(r'$\eta(t)/\eta_0$')
        plt.axhline(y=1.0, ls='--', color='k')

        ref_0 = calc_compression_ratio(
            x=np.zeros((len(lattice), len(lattice[0]), simsetup['N']), dtype=int), method='manual',
            eta_0=data_dict['compressibility_full'][0,2], datatype='full', elemtype=np.int)
        ref_1 = calc_compression_ratio(
            x=np.ones((len(lattice), len(lattice[0]), simsetup['N']), dtype=int), method='manual',
            eta_0=data_dict['compressibility_full'][0,2], datatype='full', elemtype=np.int)
        plt.axhline(y=ref_0[0], ls='-.', color='gray')
        plt.axhline(y=ref_1[0], ls='-.', color='blue')
        print(ref_0,ref_0,ref_0,ref_0, 'is', ref_0, 'vs', ref_1)
        plt.xlabel(r'$t$ (lattice steps)')
        plt.ylim(-0.05, 1.01)
        plt.savefig(
            io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_comp_remove%.2f_exo%.2f.png' %
            (exosome_string, buildstring, gridsize, num_steps, exosome_remove_ratio, field_signal_strength))
        plt.clf()  # plt.show()

    print("\nMulticell simulation complete - output in %s" % io_dict['basedir'])
    return lattice, data_dict, io_dict


if __name__ == '__main__':
    curated = False
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=random_mem, random_W=random_W, curated=curated, housekeeping=0)

    # setup: lattice sim core parameters
    n = 10                # global GRIDSIZE
    steps = 40            # global NUM_LATTICE_STEPS
    buildstring = "dual"  # init condition: mono/dual/memory_sequence/random
    meanfield = False     # True: infinite signal distance (no neighbor search; track mean field)
    plot_period = 1
    state_int = True
    beta = 2000.00        # 2.0

    # setup: signalling field (exosomes + cell-cell signalling via W matrix)
    exosome_string = "no_exo_field"   # on/off/all/no_exo_field; 'off' = send info only 'off' genes
    fieldprune = 0.0                  # amount of exo field idx to randomly prune from each cell
    field_signal_strength = 20.0      #  / (n*n) * 8   # global GAMMA = field_strength_signal tunes exosomes AND sent field

    # setup: applied/manual field (part 1)
    #field_applied = construct_app_field_from_genes(
    #    IPSC_EXTENDED_GENES_EFFECTS, simsetup['GENE_ID'], num_steps=steps)  # size N x steps or None
    field_applied = None
    field_applied_strength = 0.0

    # setup: applied/manual field (part 2) -- optionally add housekeeping field with strength Kappa
    flag_housekeeping = False
    field_housekeeping_strength = 0.0  # aka Kappa
    assert not flag_housekeeping
    if flag_housekeeping:
        assert field_housekeeping_strength > 0
        # housekeeping auto (via model extension)
        field_housekeeping = np.zeros(simsetup['N'])
        if simsetup['K'] > 0:
            field_housekeeping[-simsetup['K']:] = 1.0
            print(field_applied)
        else:
            print('Note gene 0 (on), 1 (on), 2 (on) are HK in A1 memories')
            print('Note gene 4 (off), 5 (on) are HK in C1 memories')
            field_housekeeping[4] = 1.0
            field_housekeeping[5] = 1.0
        if field_applied is not None:
            field_applied += field_housekeeping_strength * field_housekeeping
        else:
            field_applied = field_housekeeping_strength * field_housekeeping
    else:
        field_housekeeping = None
        field_housekeeping_strength = 0.0

    mc_sim_wrapper(
        simsetup, gridsize=n, num_steps=steps, buildstring=buildstring,
        exosome_string=exosome_string, exosome_remove_ratio=fieldprune,
        field_signal_strength=field_signal_strength,
        field_applied=field_applied,  field_applied_strength=field_applied_strength,
        flag_housekeeping=flag_housekeeping, field_housekeeping_strength=field_housekeeping_strength,
        beta=beta, plot_period=plot_period, state_int=state_int,
        meanfield=meanfield)
    """
    for beta in [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 4.0, 5.0, 10.0, 100.0]:
        mc_sim_wrapper(simsetup, gridsize=n, num_steps=steps, buildstring=buildstring, exosome_string=fieldstring,
               field_remove_ratio=fieldprune, ext_field_strength=ext_field_strength, app_field=app_field,
               app_field_strength=app_field_strength, beta=beta, plot_period=plot_period, state_int=state_int, meanfield=meanfield)
    """
