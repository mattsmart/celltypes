import numpy as np

from cytokine_settings import build_intracell_model, DEFAULT_CYTOKINE_MODEL, APP_FIELD_STRENGTH, RUNS_SUBDIR_CYTOKINES, BETA_CYTOKINE
from cytokine_simulate import cytokine_sim

from singlecell.singlecell_class import Cell
from singlecell.singlecell_constants import NUM_STEPS, BETA
from utils.file_io import run_subdir_setup, runinfo_append
from singlecell.singlecell_functions import state_to_label, label_to_state


def state_landscape(model_name=DEFAULT_CYTOKINE_MODEL, iterations=NUM_STEPS, applied_field_strength=APP_FIELD_STRENGTH,
                    external_field=None, flag_write=False):

    spin_labels, intxn_matrix, applied_field_const, init_state = build_intracell_model(model_name=DEFAULT_CYTOKINE_MODEL)
    N = len(spin_labels)

    labels_to_states = {idx:label_to_state(idx, N) for idx in range(2 ** N)}
    states_to_labels = {tuple(v): k for k, v in labels_to_states.items()}

    for state_label in range(2**N):
        init_cond = labels_to_states[state_label]
        print("\n\nSimulating with init state label", state_label, ":", init_cond)
        state_array, dirs = cytokine_sim(iterations=iterations, beta=BETA_CYTOKINE, flag_write=False,
                                         applied_field_strength=applied_field_strength, external_field=external_field,
                                         init_state_force=init_cond)
        label_timeseries = [states_to_labels[tuple(state_array[:,t])] for t in range(iterations)]
        for elem in label_timeseries:
            print(elem, "|", end=' ')
    return


if __name__ == '__main__':
    # For model A:
    # - deterministic oscillations between state 0 (all-off) and state 15 (all-on)
    # - if sufficient field is added, the oscillations disappear and its just stuck in the all-on state 15
    # - threshold h_0 strength is cancelling the negative feedback term J_2on0 = J[0,2] of SOCS (s_2) on R (s_0)
    # - TODO: issue seen in multicell may 10 that SOCS off => R on, logical wiring problem... need to resolve
    external_field = np.array([1,0,0,0])
    state_landscape(iterations=20, applied_field_strength=1.0, external_field=external_field)
