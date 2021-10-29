import numpy as np


# model settings
DEFAULT_CYTOKINE_MODEL = 'B'
VALID_CYTOKINE_MODELS = ['A', 'B']

# inverse temperature
BETA_CYTOKINE = 100.0

# J_ij and h_i parameters
APP_FIELD_STRENGTH = 1.0
INTXN_INTERCELL = 0.3
INTXN_AUTOCRINE = 0.5
INTXN_MEDIUM = 1.5
INTXN_PRODUCE = 1.5
BIAS_DEGRADE = INTXN_PRODUCE / 2

# io settings
RUNS_SUBDIR_CYTOKINES = "cytokines"


def build_intracell_model(model_name=DEFAULT_CYTOKINE_MODEL):

    if model_name == 'A':
        spin_labels = ["bound_dimeric_receptor",
                       "pSTAT",
                       "SOCS",
                       "cytokine"]
        # effect of each on "bound_dimeric_receptor"
        J_1_on_0 = 0.0
        J_2_on_0 = -1 * INTXN_MEDIUM  # ON SOCS => OFF bound_dimeric_receptor
        J_3_on_0 = INTXN_AUTOCRINE  # ON cytokine => ON bound receptor
        # effect of each on "pSTAT"
        J_0_on_1 = INTXN_MEDIUM  # ON bound_dimeric_receptor => ON pSTAT
        J_2_on_1 = 0.0
        J_3_on_1 = 0.0
        # effect of each on "SOCS"
        J_0_on_2 = 0.0
        J_1_on_2 = INTXN_MEDIUM  # ON pSTAT => ON SOCS
        J_3_on_2 = 0.0
        # effect of each on "cytokine"
        J_0_on_3 = 0.0
        J_1_on_3 = INTXN_MEDIUM  # ON pSTAT => ON cytokine
        J_2_on_3 = 0.0
        # fill in interaction matrix J
        intxn_matrix = np.array([[0.0,      J_1_on_0, J_2_on_0, J_3_on_0],
                                 [J_0_on_1, 0.0,      J_2_on_1, J_3_on_1],
                                 [J_0_on_2, J_1_on_2, 0.0,      J_3_on_2],
                                 [J_0_on_3, J_1_on_3, J_2_on_3, 0.0]])
        init_state = np.array([-1, -1, -1, -1])  # all off to start
        applied_field_const = np.array([1, 0, 0, 0])

    elif model_name == 'B':
        spin_labels = ["bound_dimeric_receptor",
                       "pSTAT",
                       "SOCS",
                       "cytokine"]
        # effect of each on "bound_dimeric_receptor"
        J_1_on_0 = 0.0
        J_2_on_0 = -1 * INTXN_MEDIUM / 2    # ON SOCS => OFF bound_dimeric_receptor
        J_3_on_0 = INTXN_AUTOCRINE / 2           # ON cytokine => ON bound receptor
        # effect of each on "pSTAT"
        J_0_on_1 = INTXN_PRODUCE  # ON bound_dimeric_receptor => ON pSTAT
        J_2_on_1 = 0.0
        J_3_on_1 = 0.0
        # effect of each on "SOCS"
        J_0_on_2 = 0.0
        J_1_on_2 = INTXN_PRODUCE  # ON pSTAT => ON SOCS
        J_3_on_2 = 0.0
        # effect of each on "cytokine"
        J_0_on_3 = 0.0
        J_1_on_3 = INTXN_PRODUCE  # ON pSTAT => ON cytokine
        J_2_on_3 = 0.0
        # fill in interaction matrix J
        intxn_matrix = np.array([[0.0,      J_1_on_0, J_2_on_0, J_3_on_0],
                                 [J_0_on_1, 0.0,      J_2_on_1, J_3_on_1],
                                 [J_0_on_2, J_1_on_2, 0.0,      J_3_on_2],
                                 [J_0_on_3, J_1_on_3, J_2_on_3, 0.0]])
        init_state = np.array([-1, -1, -1, -1])  # all off to start
        # everything is biased to the off state
        eps = 0.1
        applied_field_const = np.array([J_2_on_0 + J_3_on_0 - eps,  # receptor bias off, negate effect of SOCS off => R on
                                        -BIAS_DEGRADE,       # degradation of pSTAT
                                        -BIAS_DEGRADE,       # degradation of SOCS
                                        -BIAS_DEGRADE])      # degradation of C

    else:
        print("Warning: invalid model name specified")
        spin_labels = None
        intxn_matrix = None
        applied_field_const = None
        init_state = None

    return spin_labels, intxn_matrix, applied_field_const, init_state


def build_intercell_model(model_name=DEFAULT_CYTOKINE_MODEL):
    spin_labels, intxn_matrix, applied_field_const, init_state = build_intracell_model(model_name=model_name)

    if model_name == "A":
        # effect of singalling element (cytokine) in neighbouring cell on each state element
        signal_3_on_0 = INTXN_INTERCELL  # ON cytokine in neighbour => ON bound_dimeric_receptor
        signal_3_on_1 = 0.0              # effect on pSTAT
        signal_3_on_2 = 0.0              # effect on SOCS
        # fill in cell A on cell B interaction matrix
        signal_matrix = np.array([[0.0, 0.0, 0.0, signal_3_on_0],
                                  [0.0, 0.0, 0.0, signal_3_on_1],
                                  [0.0, 0.0, 0.0, signal_3_on_2],
                                  [0.0, 0.0, 0.0, 0.0]])

    else:
        print("Warning: invalid model name specified")
        signal_matrix = None

    return spin_labels, intxn_matrix, applied_field_const, init_state, signal_matrix
