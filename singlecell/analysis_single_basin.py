import matplotlib.pyplot as plt
import numpy as np
import os

from singlecell.singlecell_constants import FLAG_PRUNE_INTXN_MATRIX, FLAG_BURST_ERRORS
from utils.file_io import RUNS_FOLDER
from singlecell.singlecell_functions import state_burst_errors, single_memory_projection_timeseries
from singlecell.singlecell_simsetup import singlecell_simsetup
from singlecell.singlecell_simulate import singlecell_sim


def single_projection_timeseries_vary_basin_init(memory_label='esc', figname='mehta2014_Fig1E.png'):
    """
    Generate and plot L trajectories:
      - evolve init cond by gradually corrupting an initial memory (flip val of subset of indices)
    See Fig1E of Mehta and Lang 2014 -- they perform this analysis for the 'esc' basin
    Return: None
    """
    # function specific settings
    analysis_subdir = "mehta1E"
    flag_prune_intxn_matrix = FLAG_PRUNE_INTXN_MATRIX
    flag_burst_errors = FLAG_BURST_ERRORS
    num_traj = 50
    # NOTE ratio_amounts = np.zeros(100) + 0.15  -- thesis said to randomly flip 15% to get IC for figure 1E
    ratio_amounts = np.linspace(0.0, 0.5, num_traj)
    # perform standard simsetup
    simsetup = singlecell_simsetup(flag_prune_intxn_matrix=flag_prune_intxn_matrix)
    # basin choice
    memory_idx = simsetup['CELLTYPE_ID'][memory_label]
    init_state_pure = simsetup['XI'][:, memory_idx]
    # additional sim settings
    num_steps = 100
    # run simulation
    proj_timeseries_array = np.zeros((num_steps, len(ratio_amounts)))
    for idx, ratio_to_flip in enumerate(ratio_amounts):
        subsample_state = state_burst_errors(init_state_pure, ratio_to_flip=ratio_to_flip)
        cellstate_array, io_dict = singlecell_sim(init_state=subsample_state, iterations=num_steps, simsetup=simsetup,
                                                  flag_burst_error=flag_burst_errors, flag_write=False,
                                                  analysis_subdir=analysis_subdir, plot_period=num_steps*2)
        proj_timeseries_array[:, idx] = single_memory_projection_timeseries(cellstate_array, memory_idx, simsetup['ETA'])[:]
    # cleanup output folders from main()
    # TODO: function to cleanup runs subdir mehta1E call here call
    # plot output
    plt.plot(range(num_steps), proj_timeseries_array, color='blue', linewidth=0.75)
    plt.title('Mehta Fig 1E analog: proj on memory %s while corrupting IC' % (memory_label))
    plt.ylabel('proj on memory %s' % (memory_label))
    plt.xlabel('Time (10^3 updates, all spins)')
    plt.savefig(RUNS_FOLDER + os.sep + analysis_subdir + os.sep + figname)
    return


if __name__ == '__main__':
    single_projection_timeseries_vary_basin_init()
