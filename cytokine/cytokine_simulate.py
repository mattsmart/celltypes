import numpy as np

from cytokine_settings import build_intracell_model, DEFAULT_CYTOKINE_MODEL, APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES, BETA_CYTOKINE
from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import NUM_STEPS
from utils.file_io import run_subdir_setup, runinfo_append


def cytokine_sim(model_name=DEFAULT_CYTOKINE_MODEL, iterations=NUM_STEPS, beta=BETA_CYTOKINE, applied_field_strength=APP_FIELD_STRENGTH,
                 external_field=None, init_state_force=None, flag_write=False, flag_print=False):

    # setup model and init cell class
    spin_labels, intxn_matrix, applied_field_const, init_state = build_intracell_model(model_name=model_name)
    if init_state_force is None:
        cell = Cell(init_state, "model_%s" % model_name, memories_list=[], gene_list=spin_labels)
    else:
        cell = Cell(init_state_force, "model_%s_init_state_forced" % model_name, memories_list=[], gene_list=spin_labels)

    # augment applied field
    if external_field is not None:
        assert np.shape(applied_field_const) == np.shape(applied_field_const)
        applied_field_const += external_field

    # io
    if flag_write:
        io_dict = run_subdir_setup(run_subfolder=RUNS_SUBDIR_CYTOKINES)
        dirs = [io_dict['basedir'], io_dict['datadir'], io_dict['latticedir'], io_dict['plotdir']]
    else:
        dirs = None

    # simulate
    for step in range(iterations-1):

        if flag_print:
            print(cell.steps, "cell steps:", cell.get_current_state(), "aka", cell.get_current_label())

        # plotting
        #if singlecell.steps % plot_period == 0:
        #    fig, ax, proj = singlecell.plot_projection(use_radar=True, pltdir=plot_lattice_folder)

        cell.update_state(intxn_matrix=intxn_matrix, beta=beta, field_applied=applied_field_const, field_applied_strength=applied_field_strength)

    # end state
    if flag_print:
        print(cell.steps, "cell steps:", cell.get_current_state(), "aka", cell.get_current_label())

    # Write
    if flag_write:
        print("Writing state to file..")
        cell.write_state(data_folder)

    if flag_print:
        print("Done")
    return cell.get_state_array(), dirs


if __name__ == '__main__':
    external_field = np.array([0,0,0,0])
    cytokine_sim(iterations=20, applied_field_strength=1.0, external_field=external_field, flag_write=True, flag_print=True)
