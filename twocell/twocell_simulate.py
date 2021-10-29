import utils.init_multiprocessing  # BEFORE numpy

import numpy as np
import os

from twocell.twocell_visualize import simple_vis, lattice_timeseries_proj_grid, lattice_timeseries_overlap, \
    lattice_timeseries_energy, lattice_timeseries_state_grid
from multicell.multicell_spatialcell import SpatialCell
from multicell.multicell_constants import VALID_EXOSOME_STRINGS
from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import MEMS_UNFOLD, BETA
from utils.file_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_functions import single_memory_projection_timeseries, hamiltonian
from singlecell.singlecell_simsetup import singlecell_simsetup


EXOSOME_STRING = 'no_exo_field'
EXOSOME_PRUNE = 0.0
PLOT_PERIOD = 10

# TODO file IO
# TODO new full timeseries visualizations
# TODO compute indiv energies, interaction term, display in plot somehow neatly?



def twocell_sim_troubleshoot(beta=200.0, gamma=0.0, flag_01=False):

    def random_twocell_lattice(simsetup):
        #cell_a_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in xrange(simsetup['N'])]).T
        #cell_b_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in xrange(simsetup['N'])]).T
        cell_a_init = np.ones(20) #np.array([-1, -1, 1, -1,-1,-1])
        cell_b_init = np.ones(20) #np.array([1, 1, 1, 1, 1, 1])
        lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                    SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]
        return lattice

    def prep_monolothic(lattice, simsetup):
        J_singlecell = simsetup['J']
        W_matrix = simsetup['FIELD_SEND']

        # build multicell Jij matrix (2N x 2N)
        numspins = 2 * simsetup['N']
        J_multicell = np.zeros((numspins, numspins))
        block_diag = J_singlecell
        block_offdiag = gamma * W_matrix
        J_multicell[0:simsetup['N'], 0:simsetup['N']] = block_diag
        J_multicell[-simsetup['N']:, -simsetup['N']:] = block_diag
        J_multicell[0:simsetup['N'], -simsetup['N']:] = block_offdiag
        J_multicell[-simsetup['N']:, 0:simsetup['N']] = block_offdiag

        # build multicell applied field vector (2N x 1)
        h_multicell = None
        if flag_01:
            h_multicell = np.zeros(numspins)
            W_dot_one_scaled = np.dot(W_matrix, np.ones(simsetup['N'])) * gamma / 2
            h_multicell[0:simsetup['N']] = W_dot_one_scaled
            h_multicell[-simsetup['N']:] = W_dot_one_scaled

        # cell setup
        init_state = np.zeros(numspins)
        init_state[0:simsetup['N']] = lattice[0][0].get_current_state()
        init_state[-simsetup['N']:] = lattice[0][1].get_current_state()
        genelabels_multicell = simsetup['GENE_LABELS'] + simsetup['GENE_LABELS']
        singlecell = Cell(init_state, 'multicell', memories_list=simsetup['CELLTYPE_LABELS'],
                          gene_list=genelabels_multicell)

        return J_multicell, h_multicell, singlecell

    def step_monolithic(J_multicell, h_multicell, singlecell, async_flag=True):
        singlecell.update_state(J_multicell, beta=beta, field_applied=h_multicell, field_applied_strength=1.0,
                                async_batch=True, async_flag=async_flag)
        return singlecell

    def step_staggered(lattice, simsetup, async_flag=True):
        cell_A = lattice[0][0]
        cell_B = lattice[0][1]

        cell_A_01_init = (cell_A.get_current_state() + 1) / 2.0
        cell_B_01_init = (cell_B.get_current_state() + 1) / 2.0

        # update cell A -- first get nbr field
        nbr_cell_state_01_rep = (cell_B.get_current_state() + 1) / 2.0  # convert to 0, 1 rep
        total_field_A = gamma * np.dot(simsetup['FIELD_SEND'], cell_B_01_init)
        print("total_field_A")
        print(total_field_A)
        cell_A.update_state(simsetup['J'], beta=beta, field_signal_strength=1.0, field_signal=total_field_A,
                            async_flag=async_flag)

        # update cell B -- first get nbr field
        nbr_cell_state_01_rep = (cell_A.get_current_state() + 1) / 2.0  # convert to 0, 1 rep
        total_field_B = gamma * np.dot(simsetup['FIELD_SEND'], cell_A_01_init)
        print("total_field_B")
        print(total_field_B)
        cell_B.update_state(simsetup['J'], beta=beta, field_signal_strength=1.0, field_signal=total_field_B,
                            async_flag=async_flag)

        return lattice


    def extract_monolithic(singlecell):
        # repackage final state as multicell lattice
        final_state = singlecell.get_current_state()
        cell_a_init = final_state[0:simsetup['N']]
        cell_b_init = final_state[-simsetup['N']:]
        lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                    SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]
        return lattice

    simsetup = singlecell_simsetup(
        unfolding=True, random_mem=False, random_W=False, npzpath=MEMS_UNFOLD, curated=True)
    lattice_staggered = random_twocell_lattice(simsetup)
    J_multicell, h_multicell, lattice_monolithic = prep_monolothic(lattice_staggered, simsetup)
    print("INIT COND")
    print(lattice_staggered[0][0].get_current_state())
    print(lattice_staggered[0][1].get_current_state())
    print(lattice_monolithic.get_current_state())

    num_steps = 3
    for step in range(num_steps):

        print("\nCURRENT STEP:", step)
        # monolthic step
        print("\nmonolithic stepping...")
        lattice_monolithic = step_monolithic(J_multicell, h_multicell, lattice_monolithic,
                                             async_flag=False)
        lattice_extracted = extract_monolithic(lattice_monolithic)
        print("print lattice_monolithic.get_current_state() (step %d)" % step)
        print(lattice_monolithic.get_current_state())
        print(lattice_extracted[0][0].get_current_state())
        print(lattice_extracted[0][1].get_current_state())

        # staggered step
        print("\nlattice_staggered stepping...")
        lattice_staggered = step_staggered(lattice_staggered, simsetup, async_flag=False)
        print("print cell_A.get_current_state() (step %d)" % step)
        print(lattice_staggered[0][0].get_current_state())
        print("print cell_B.get_current_state() (step %d)" % step)
        print(lattice_staggered[0][1].get_current_state())

    print("\nendproj...")
    XI_scaled = simsetup['XI'] / simsetup['N']
    print("MONOLITHIC:")
    cell_A_endstate = lattice_extracted[0][0].get_state_array()[:, -1]
    cell_B_endstate = lattice_extracted[0][1].get_state_array()[:, -1]
    cell_A_overlaps = np.dot(XI_scaled.T, cell_A_endstate)
    cell_B_overlaps = np.dot(XI_scaled.T, cell_B_endstate)
    print(cell_A_overlaps, cell_B_overlaps)
    print("STAGGERED:")
    cell_A_endstate = lattice_staggered[0][0].get_state_array()[:, -1]
    cell_B_endstate = lattice_staggered[0][1].get_state_array()[:, -1]
    cell_A_overlaps = np.dot(XI_scaled.T, cell_A_endstate)
    cell_B_overlaps = np.dot(XI_scaled.T, cell_B_endstate)
    print(cell_A_overlaps, cell_B_overlaps)

    return


def twocell_sim_as_onelargemodel(lattice, simsetup, num_steps, beta=BETA, gamma=1.0,
                                 async_flag=True, flag_01=False):

    J_singlecell = simsetup['J']
    W_matrix = simsetup['FIELD_SEND']

    # build multicell Jij matrix (2N x 2N)
    numspins = 2*simsetup['N']
    J_multicell = np.zeros((numspins, numspins))
    block_diag = J_singlecell
    block_offdiag = gamma * W_matrix
    J_multicell[0:simsetup['N'], 0:simsetup['N']] = block_diag
    J_multicell[-simsetup['N']:, -simsetup['N']:] = block_diag
    J_multicell[0:simsetup['N'], -simsetup['N']:] = block_offdiag
    J_multicell[-simsetup['N']:, 0:simsetup['N']] = block_offdiag

    # build multicell applied field vector (2N x 1)
    h_multicell = None
    if flag_01:
        h_multicell = np.zeros(numspins)
        W_dot_one_scaled = np.dot(W_matrix, np.ones(simsetup['N'])) * gamma / 2
        h_multicell[0:simsetup['N']] = W_dot_one_scaled
        h_multicell[-simsetup['N']:] = W_dot_one_scaled

    # cell setup
    init_state = np.zeros(numspins)
    init_state[0:simsetup['N']] = lattice[0][0].get_current_state()
    init_state[-simsetup['N']:] = lattice[0][1].get_current_state()
    genelabels_multicell = simsetup['GENE_LABELS'] + simsetup['GENE_LABELS']
    singlecell = Cell(init_state, 'multicell', memories_list=simsetup['CELLTYPE_LABELS'],
                      gene_list=genelabels_multicell)

    # simulate
    for step in range(num_steps):
        singlecell.update_state(J_multicell, beta=beta, field_applied=h_multicell, field_applied_strength=1.0,
                                async_flag=async_flag)

    # repackage final state as multicell lattice
    final_state = singlecell.get_current_state()
    cell_a_init = final_state[0:simsetup['N']]
    cell_b_init = final_state[-simsetup['N']:]
    lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]
    return lattice


def twocell_sim_fast(lattice, simsetup, num_steps, beta=BETA, gamma=1.0, field_applied=None,
                     field_applied_strength=0.0, async_flag=True, flag_01=False):
    cell_A = lattice[0][0]
    cell_B = lattice[0][1]
    for step in range(num_steps):
        if async_flag:
            # update cell A -- first get nbr field
            cell_B_time_t = cell_B.get_current_state()
            if flag_01:
                cell_B_sent_time_t = (cell_B_time_t + 1) / 2.0  # convert to 0, 1 rep for biological dot product below
            else:
                cell_B_sent_time_t = cell_B_time_t
            total_field_A = gamma * np.dot(simsetup['FIELD_SEND'], cell_B_sent_time_t)
            cell_A.update_state(simsetup['J'], beta=beta, field_signal_strength=1.0, field_applied=None,
                                field_signal=total_field_A, async_flag=async_flag)
            # update cell B -- first get nbr field
            cell_A_time_t = cell_A.get_current_state()
            if flag_01:
                cell_A_sent_time_t = (cell_A_time_t + 1) / 2.0  # convert to 0, 1 rep for biological dot product below
            else:
                cell_A_sent_time_t = cell_A_time_t
            total_field_B = gamma * np.dot(simsetup['FIELD_SEND'], cell_A_sent_time_t)
            cell_B.update_state(simsetup['J'], beta=beta, field_signal_strength=1.0, field_applied=None,
                                field_signal=total_field_B, async_flag=async_flag)

        else:
            cell_A_time_t = cell_A.get_current_state()
            cell_B_time_t = cell_B.get_current_state()
            if flag_01:
                cell_A_sent_time_t = (cell_A_time_t + 1) / 2.0  # convert to 0, 1 rep for biological dot product below
                cell_B_sent_time_t = (cell_B_time_t + 1) / 2.0  # convert to 0, 1 rep for biological dot product below
            else:
                cell_A_sent_time_t = cell_A_time_t
                cell_B_sent_time_t = cell_B_time_t
            # update cell A -- first get nbr field
            total_field_A = gamma * np.dot(simsetup['FIELD_SEND'], cell_B_sent_time_t)
            cell_A.update_state(
                simsetup['J'], beta=beta, field_signal_strength=1.0, field_applied=None,
                field_signal=total_field_A, async_flag=async_flag)
            # update cell B -- first get nbr field
            total_field_B = gamma * np.dot(simsetup['FIELD_SEND'], cell_A_sent_time_t)
            cell_B.update_state(
                simsetup['J'], beta=beta, field_signal_strength=1.0, field_applied=None,
                field_signal=total_field_B, async_flag=async_flag)
    return lattice


def twocell_sim(lattice, simsetup, num_steps, data_dict, io_dict, beta=BETA,
                exostring=EXOSOME_STRING, exoprune=EXOSOME_PRUNE, gamma=1.0,
                field_applied=None, field_applied_strength=0.0, ioflag=True):

    cell_A = lattice[0][0]
    cell_B = lattice[0][1]
    # local fields initialization
    neighbours_A = [[0, 1]]
    neighbours_B = [[0, 0]]
    # initial condition vis
    if ioflag:
        simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Initial condition', savemod='_%d' % 0)
    for step in range(num_steps-1):
        # TODO could compare against whole model random update sequence instead of this block version
        field_applied_step = field_applied  # TODO housekeeping applied field; N vs N+M
        # update cell A
        total_field_A, _ = cell_A.get_local_exosome_field(
            lattice, None, None, exosome_string=exostring,
            exosome_remove_ratio=exoprune, neighbours=neighbours_A)
        if simsetup['FIELD_SEND'] is not None:
            total_field_A += cell_A.get_local_paracrine_field(lattice, neighbours_A, simsetup)
        cell_A.update_state(simsetup['J'], beta=beta,
                            field_signal=total_field_A,
                            field_signal_strength=gamma,
                            field_applied=field_applied_step,
                            field_applied_strength=field_applied_strength)
        if ioflag and num_steps % PLOT_PERIOD == 0:
            simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Step %dA' % step, savemod='_%dA' % step)
        # update cell B
        total_field_B, _ = cell_B.get_local_exosome_field(lattice, None, None, exosome_string=exostring,
                                                          exosome_remove_ratio=exoprune, neighbours=neighbours_B)
        if simsetup['FIELD_SEND'] is not None:
            total_field_B += cell_B.get_local_paracrine_field(lattice, neighbours_B, simsetup)
        cell_B.update_state(simsetup['J'], beta=beta,
                            field_signal=total_field_B,
                            field_signal_strength=gamma,
                            field_applied=field_applied_step,
                            field_applied_strength=field_applied_strength)
        if ioflag and num_steps % PLOT_PERIOD == 0:
            simple_vis(lattice, simsetup, io_dict['plotlatticedir'], 'Step %dB' % step, savemod='_%dB' % step)

    # fill in data
    print('simulation done; gathering data')
    if 'memory_proj_arr' in list(data_dict.keys()):
        for memory_idx in range(simsetup['P']):
            data_dict['memory_proj_arr'][memory_idx][0, :] = \
                single_memory_projection_timeseries(cell_A.get_state_array(), memory_idx, simsetup['ETA'])
            data_dict['memory_proj_arr'][memory_idx][1, :] = \
                single_memory_projection_timeseries(cell_B.get_state_array(), memory_idx, simsetup['ETA'])
    if 'overlap' in list(data_dict.keys()):
        data_dict['overlap'] = np.array([
            np.dot(cell_A.state_array[:, i], cell_B.state_array[:, i]) for i in range(num_steps)]) / simsetup['N']
    if 'grid_state_int' in list(data_dict.keys()):
        # TODO
        print('TODO grid_state_int data fill in')
        #data_dict['grid_state_int'] = np.zeros((2, num_steps), dtype=int)
    if 'multi_hamiltonian' in list(data_dict.keys()):
        data_dict['single_hamiltonians'][0, :] = [hamiltonian(cell_A.state_array[:, i], simsetup['J'], field=field_applied, fs=field_applied_strength) for i in range(num_steps)]
        data_dict['single_hamiltonians'][1, :] = [hamiltonian(cell_B.state_array[:, i], simsetup['J'], field=field_applied, fs=field_applied_strength) for i in range(num_steps)]
        if simsetup['FIELD_SEND'] is not None:
            # TODO check the algebra here...
            W = simsetup['FIELD_SEND']
            WdotOne = np.dot(W, np.ones(simsetup['N']))
            WSym2 = (W + W.T)
            data_dict['unweighted_coupling_term'][:] = \
                [- 0.5 * np.dot( cell_A.state_array[:, i], np.dot(WSym2, cell_B.state_array[:, i]) )
                 - 0.5 * np.dot(WdotOne, cell_A.state_array[:, i] + cell_B.state_array[:, i])
                 for i in range(num_steps)]
        data_dict['multi_hamiltonian'] = (data_dict['single_hamiltonians'][0, :] + data_dict['single_hamiltonians'][1, :]) + gamma * data_dict['unweighted_coupling_term']
    return lattice, data_dict, io_dict


def twocell_simprep(simsetup, num_steps, beta=BETA, exostring=EXOSOME_STRING, exoprune=EXOSOME_PRUNE, gamma=1.0,
                    field_applied=None, field_applied_strength=0.0):
    """
    Prep lattice (of two cells), fields, and IO
    """
    # check args
    assert type(num_steps) is int
    assert exostring in VALID_EXOSOME_STRINGS
    assert 0.0 <= exoprune < 1.0
    assert 0.0 <= gamma < 10.0

    cell_a_init = simsetup['XI'][:, 0]
    cell_b_init = simsetup['XI'][:, 0]
    lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]  # list of list to conform to multicell slattice funtions

    # app fields initialization
    field_applied_step = None  # TODO housekeeping applied field; N vs N+M

    # setup io dict
    io_dict = run_subdir_setup()
    info_list = [['memories_path', simsetup['memories_path']],
                 ['script', 'twocell_simulate.py'],
                 ['num_steps', num_steps],
                 ['exosome_string', exostring],
                 ['exosome_remove_ratio', exoprune],
                 ['field_applied_strength', field_applied_strength],
                 ['field_signal_strength', gamma],
                 ['field_applied', field_applied],
                 ['beta', beta],
                 ['random_mem', simsetup['random_mem']],
                 ['random_W', simsetup['random_W']]]
    runinfo_append(io_dict, info_list, multi=True)
    # conditionally store random mem and W
    np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_XI.txt', simsetup['XI'], delimiter=',', fmt='%d')
    if simsetup['FIELD_SEND'] is not None:
        np.savetxt(io_dict['simsetupdir'] + os.sep + 'simsetup_W.txt', simsetup['FIELD_SEND'], delimiter=',', fmt='%.4f')
    else:
        runinfo_append(io_dict, [simsetup['FIELD_SEND'], None], multi=False)

    # setup data dictionary
    # TODO increase timstep resolution by x2 -- every cell update (see sequence info...)
    data_dict = {}
    store_state_int = True
    store_memory_proj_arr = True
    store_overlap = True
    store_energy = True
    # TODO what data are we storing?
    # store projection onto each memory
    if store_memory_proj_arr:
        data_dict['memory_proj_arr'] = {}
        for idx in range(simsetup['P']):
            data_dict['memory_proj_arr'][idx] = np.zeros((2, num_steps))
    # store cell-cell overlap as scalar
    if store_overlap:
        data_dict['overlap'] = np.zeros(num_steps)
    # store state as int (compressed)
    if store_state_int:
        assert simsetup['N'] < 12
        data_dict['grid_state_int'] = np.zeros((2, num_steps), dtype=int)
    if store_energy:
        data_dict['single_hamiltonians'] = np.zeros((2, num_steps))
        data_dict['multi_hamiltonian'] = np.zeros(num_steps)
        data_dict['unweighted_coupling_term'] = np.zeros(num_steps)

    # run the simulation
    lattice, data_dict, io_dict = \
        twocell_sim(lattice, simsetup, num_steps, data_dict, io_dict, beta=beta,
                    exostring=exostring, exoprune=exoprune, gamma=gamma,
                    field_applied=field_applied, field_applied_strength=field_applied_strength)

    # check the data dict
    """
    for data_idx, memory_idx in enumerate(data_dict['memory_proj_arr'].keys()):
        print data_dict['memory_proj_arr'][memory_idx]
        plt.plot(data_dict['memory_proj_arr'][memory_idx].T)
        plt.ylabel('Projection of all cells onto type: %s' % simsetup['CELLTYPE_LABELS'][memory_idx])
        plt.xlabel('Time (full lattice steps)')
        plt.savefig(io_dict['plotdatadir'] + os.sep + '%s_%s_n%d_t%d_proj%d_remove%.2f_exo%.2f.png' %
                    (exosome_string, buildstring, gridsize, num_steps, memory_idx, field_remove_ratio, ext_field_strength))
        plt.clf()  #plt.show()
    """
    # additional visualizations
    lattice_timeseries_state_grid(lattice, simsetup, io_dict['plotdatadir'], savemod='')
    lattice_timeseries_proj_grid(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    lattice_timeseries_proj_grid(data_dict, simsetup, io_dict['plotdatadir'], savemod='conj', conj=True)
    lattice_timeseries_overlap(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    # TODO include app field energy
    lattice_timeseries_energy(data_dict, simsetup, io_dict['plotdatadir'], savemod='')
    return lattice, data_dict, io_dict


if __name__ == '__main__':
    """
    HOUSEKEEPING = 0
    KAPPA = 100

    random_mem = False
    random_W = False
    #simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA)
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   housekeeping=HOUSEKEEPING)
    print 'note: N =', simsetup['N']
    steps = 20
    beta = 2.0  # 2.0

    exostring = "no_exo_field"  # on/off/all/no_exo_field, note e.g. 'off' means send info about 'off' genes only
    exoprune = 0.0              # amount of exosome field idx to randomly prune from each cell
    gamma = 0.0                 # global FIELD_SIGNAL_STRENGTH tunes exosomes AND sent field

    app_field = None
    if KAPPA > 0 and HOUSEKEEPING > 0:
        app_field = np.zeros(simsetup['N'])
        app_field[-HOUSEKEEPING:] = 1.0

    lattice, data_dict, io_dict = \
        twocell_simprep(simsetup, steps, beta=beta, exostring=exostring, exoprune=exoprune, gamma=gamma,
                        app_field=app_field, app_field_strength=KAPPA)
    """
    twocell_sim_troubleshoot(gamma=10000.0)
