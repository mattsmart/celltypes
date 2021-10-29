import numpy as np
import os
import random
import matplotlib.pyplot as plt

from cytokine_lattice_build import build_cytokine_lattice_mono
from cytokine_settings import APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES
from singlecell.singlecell_functions import state_to_label
from utils.file_io import run_subdir_setup, runinfo_append
from multicell.multicell_lattice import get_cell_locations, write_state_all_cells, printer_labels


GRIDSIZE = 4
NUM_LATTICE_STEPS = 5
SEARCH_RADIUS_CELL = 1  # TODO find nice way to have none flag here for inf range singles / homogeneous?


def run_cytokine_network(lattice, num_lattice_steps, intxn_matrix, signal_matrix, app_field_const=None, app_field_strength=APP_FIELD_STRENGTH, flag_write=False):

    # Input checks
    N = len(intxn_matrix)
    n = len(lattice)
    assert n == len(lattice[0])  # work with square lattice for simplicity
    if app_field_const is not None:
        assert len(app_field_const) == N
        app_field_timestep = app_field_const
    else:
        app_field_timestep = None

    # io
    if flag_write:
        io_dict = run_subdir_setup(run_subfolder=RUNS_SUBDIR_CYTOKINES)
        dirs = [io_dict['basedir'], io_dict['datadir'], io_dict['latticedir'], io_dict['plotdir']]
    else:
        dirs = None

    cell_locations = get_cell_locations(lattice, n)
    loc_to_idx = {pair: idx for idx, pair in enumerate(cell_locations)}

    # plot initial state of the lattice
    """
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, plot_lattice_folder, mem_idx)
    """

    # initial condition plot
    printer_labels(lattice)
    """
    graph_lattice_projection_composite(lattice, 0, n, plot_lattice_folder)
    graph_lattice_reference_overlap_plotter(lattice, 0, n, plot_lattice_folder)
    if flag_uniplots:
        for mem_idx in memory_idx_list:
            lattice_uniplotter(lattice, 0, n, plot_lattice_folder, mem_idx)
    """

    for turn in range(1, num_lattice_steps):
        print('Turn ', turn)
        random.shuffle(cell_locations)
        for idx, loc in enumerate(cell_locations):
            cell = lattice[loc[0]][loc[1]]
            cell.update_with_signal_field(lattice, SEARCH_RADIUS_CELL, n, beta=100.0, intxn_matrix=intxn_matrix,
                                          signal_matrix=signal_matrix, exosome_string="no_exo_field",
                                          field_app=app_field_timestep, field_app_strength=app_field_strength,
                                          field_signal_strength=1.0)
            #print "Cell at", loc, "is in state:", state_to_label(tuple(cell.get_current_state()))

            """
            if turn % (40*plot_period) == 0:  # plot proj visualization of each cell (takes a while; every k lat plots)
                fig, ax, proj = cell.plot_projection(use_radar=False, pltdir=plot_lattice_folder)
            """

        printer_labels(lattice)
        """
        if turn % plot_period == 0:  # plot the lattice
            graph_lattice_projection_composite(lattice, turn, n, plot_lattice_folder)
            graph_lattice_reference_overlap_plotter(lattice, turn, n, plot_lattice_folder)
            if flag_uniplots:
                for mem_idx in memory_idx_list:
                    lattice_uniplotter(lattice, turn, n, plot_lattice_folder, mem_idx)
        """

    return lattice, dirs


def wrapper_cytokine_network(gridsize=GRIDSIZE, num_steps=NUM_LATTICE_STEPS, init_cond_force=None, app_field_strength=APP_FIELD_STRENGTH, external_field=None, flag_write=False):

    # setup lattice IC
    lattice, spin_labels, intxn_matrix, applied_field_const, init_state, signal_matrix = build_cytokine_lattice_mono(gridsize, init_cond_force=init_cond_force)

    # augment applied field
    if external_field is not None:
        assert np.shape(applied_field_const) == np.shape(applied_field_const)
        applied_field_const += external_field

    # run the simulation
    lattice, dirs = run_cytokine_network(lattice, num_steps, intxn_matrix, signal_matrix, app_field_const=applied_field_const,
                                         app_field_strength=app_field_strength, flag_write=flag_write)

    # write cell state TODO: and data_dict to file
    #write_state_all_cells(lattice, data_folder)

    print("\nMulticell simulation complete")
    if flag_write:
        print("output in %s" % dirs[0])
    return lattice, dirs


if __name__ == '__main__':
    init_cond = np.array([1,-1,-1,-1])
    external_field = np.array([1,0,0,0])
    wrapper_cytokine_network(init_cond_force=init_cond, external_field=external_field)
