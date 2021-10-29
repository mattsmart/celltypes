import utils.init_multiprocessing  # BEFORE numpy

import numpy as np
import os
import matplotlib.pyplot as plt

from twocell.twocell_simulate import twocell_sim_fast, twocell_sim_as_onelargemodel
from multicell.multicell_spatialcell import SpatialCell
from singlecell.singlecell_constants import MEMS_UNFOLD, BETA
from utils.file_io import RUNS_FOLDER
from singlecell.singlecell_simsetup import singlecell_simsetup


def twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=10, monolothic_flag=False):
    # TODO issue: the monolothic sim (big Jij) has different statistics than the cell-by-cell sim
    # TODO        more differences than one would expect -- can only see for nonzero gamma
    # TODO        CHECK: behaviour of each approach for one low temp traj at high gamma
    # TODO also note that any asymmetry (on the x=y reflection line) in the mA vs mB scatterplot is unexpected

    overlap_data = np.zeros((ens, 2 * simsetup['P']))
    assert simsetup['P'] == 1

    XI_scaled = simsetup['XI'] / simsetup['N']

    def random_twocell_lattice():
        cell_a_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in range(simsetup['N'])]).T
        cell_b_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in range(simsetup['N'])]).T
        #cell_a_init = np.ones(20) #np.array([-1, -1, 1, -1,-1,-1])
        #cell_b_init = np.ones(20) #np.array([1, 1, 1, 1, 1, 1])
        lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                    SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]
        return lattice

    for traj in range(ens):
        if traj % 100 == 0:
            print("Running traj", traj, "...")
        lattice = random_twocell_lattice()

        # TODO replace with twocell_sim_as_onelargemodel (i.e. one big ising model)
        if monolothic_flag:
            lattice = twocell_sim_as_onelargemodel(
                lattice, simsetup, steps, beta=beta, gamma=gamma, async_flag=False)
        else:
            lattice = twocell_sim_fast(
                lattice, simsetup, steps, beta=beta, gamma=gamma, async_flag=False)
        cell_A_endstate = lattice[0][0].get_state_array()[:, -1]
        cell_B_endstate = lattice[0][1].get_state_array()[:, -1]
        cell_A_overlaps = np.dot(XI_scaled.T, cell_A_endstate)
        cell_B_overlaps = np.dot(XI_scaled.T, cell_B_endstate)
        overlap_data[traj, 0:simsetup['P']] = cell_A_overlaps
        overlap_data[traj, simsetup['P']:] = cell_B_overlaps

    if simsetup['P'] == 1:
        plt.figure()
        plt.scatter(overlap_data[:,0], overlap_data[:,1], alpha=0.2)
        fname = "overlaps_ens%d_beta%.2f_gamma%.2f_mono%d.png" % (ens, beta, gamma, monolothic_flag)
        plt.title(fname)
        plt.xlabel(r"$m_A$")
        plt.ylabel(r"$m_B$")
        plt.xlim(-1.05, 1.05)
        plt.ylim(-1.05, 1.05)
        plt.savefig(RUNS_FOLDER + os.sep + "twocell_analysis" + os.sep + fname)
        plt.close()
        print(fname)
        """
        import seaborn as sns; sns.set()
        import pandas as pd
        df_overlap_data = pd.DataFrame({r"$m_A$":overlap_data[:,0], r"$m_B$":overlap_data[:,1]})
        
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        #ax = sns.scatterplot(x=r"$m_A$", y=r"$m_B$", palette=cmap,
        #                     sizes=(20, 200), hue_norm=(0, 7), legend="full", data=df_overlap_data)
        ax = sns.kdeplot(overlap_data[:,0], overlap_data[:,1], shade=True, palette=cmap)
        plt.show()
        """

    else:
        # TODO do some dim reduction
        assert 1==2
    return overlap_data


def twocell_coarse_hamiltonian(simsetup, gamma, ens=10000):

    def random_twocell_lattice():
        cell_a_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in range(simsetup['N'])]).T
        cell_b_init = np.array([2*int(np.random.rand() < .5) - 1 for _ in range(simsetup['N'])]).T
        lattice = [[SpatialCell(cell_a_init, 'Cell A', [0, 0], simsetup),
                    SpatialCell(cell_b_init, 'Cell B', [0, 1], simsetup)]]
        return lattice

    energy_data = np.zeros((ens, 3))
    assert simsetup['P'] == 1

    XI_scaled = simsetup['XI'] / simsetup['N']
    W_matrix = simsetup['FIELD_SEND']
    W_matrix_sym = 0.5 * (W_matrix + W_matrix.T)
    W_dot_one_scaled = np.dot(W_matrix, np.ones(simsetup['N'])) * gamma / 2

    """
    for elem in xrange(ens):
        lattice = random_twocell_lattice()
        cell_a = lattice[0][0].get_current_state()
        cell_b = lattice[0][1].get_current_state()

        energy_A = -0.5 * np.dot(cell_a, np.dot(simsetup['J'], cell_a))
        energy_B = -0.5 * np.dot(cell_b, np.dot(simsetup['J'], cell_b))
        energy_coupling = - np.dot(W_dot_one_scaled, cell_a + cell_b) - 0.5 * np.dot( cell_a, np.dot(W_matrix, cell_b) ) - 0.5 * np.dot( cell_b, np.dot(W_matrix, cell_a) )

        energy_data[elem, 0] = np.dot(XI_scaled.T, cell_a)
        energy_data[elem, 1] = np.dot(XI_scaled.T, cell_b)
        energy_data[elem, 2] = energy_A + energy_B + energy_coupling
    """

    lattice = random_twocell_lattice()

    def beta_anneal(elem):
        assert ens > 1000
        timer = elem % 100
        beta_low = 0.01
        beta_mid = 1.5
        beta_high = 20.0
        # want low for around 50 steps, high for around 50 steps
        if timer <= 33:
            beta_step = beta_low
        elif 33 < timer <= 90:
            beta_step = beta_mid
        else:
            beta_step = beta_high
        return beta_step

    for elem in range(ens):

        # anneal to reach the corners
        beta_schedule = beta_anneal(elem)
        lattice = twocell_sim_as_onelargemodel(
            lattice, simsetup, steps, beta=beta_schedule, gamma=0.0, async_flag=False)

        cell_a = lattice[0][0].get_current_state()
        cell_b = lattice[0][1].get_current_state()
        print(elem, beta_schedule, np.dot(cell_a, np.ones(20)) / 20.0, np.dot(cell_b, np.ones(20)) / 20.0)

        energy_A = -0.5 * np.dot(cell_a, np.dot(simsetup['J'], cell_a))
        energy_B = -0.5 * np.dot(cell_b, np.dot(simsetup['J'], cell_b))
        energy_coupling = - np.dot(W_dot_one_scaled, cell_a + cell_b) - 0.5 * np.dot( cell_a, np.dot(W_matrix, cell_b) ) - 0.5 * np.dot( cell_b, np.dot(W_matrix, cell_a) )

        energy_data[elem, 0] = np.dot(XI_scaled.T, cell_a)
        energy_data[elem, 1] = np.dot(XI_scaled.T, cell_b)
        energy_data[elem, 2] = energy_A + energy_B + energy_coupling

    # plot alt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(energy_data[:, 0], energy_data[:, 1], energy_data[:, 2], c=energy_data[:, 2], marker='o')
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.show()
    plt.close()

    # plotting
    import seaborn as sns
    sns.set()
    import pandas as pd
    emax = np.max(energy_data[:, 2])
    emin = np.min(energy_data[:, 2])
    df_overlap_data = pd.DataFrame({r"$m_A$": energy_data[:, 0],
                                    r"$m_B$": energy_data[:, 1],
                                    r'$H(s)$': energy_data[:, 2],
                                    'normed_energy': (emax - emin)/(energy_data[:, 2] - emin)})

    plt.figure()
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True, reverse=True)
    ax = sns.scatterplot(x=r"$m_A$", y=r"$m_B$", palette=cmap, legend='brief',
                         hue=r'$H(s)$', data=df_overlap_data)
    fname = "energyrough_ens%d_gamma%.2f.pdf" % (ens, gamma)
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.savefig(RUNS_FOLDER + os.sep + "twocell_analysis" + os.sep + fname)
    plt.close()

    return

if __name__ == '__main__':
    random_mem = False
    random_W = False
    simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, npzpath=MEMS_UNFOLD,
                                   curated=True)
    print('note: N =', simsetup['N'])

    ensemble = 2500
    steps = 10
    beta_low = 1.5  # 2.0
    beta_mid = 10.0
    beta_high = 100.0
    gamma = 20.0
    twocell_coarse_hamiltonian(simsetup, gamma, ens=5000)
    twocell_ensemble_stats(simsetup, steps, beta_low, gamma, ens=ensemble, monolothic_flag=False)
    twocell_ensemble_stats(simsetup, steps, beta_low, gamma, ens=ensemble, monolothic_flag=True)
    twocell_ensemble_stats(simsetup, steps, beta_mid, gamma, ens=ensemble, monolothic_flag=False)
    twocell_ensemble_stats(simsetup, steps, beta_mid, gamma, ens=ensemble, monolothic_flag=True)
    twocell_ensemble_stats(simsetup, steps, beta_high, gamma, ens=ensemble, monolothic_flag=False)
    twocell_ensemble_stats(simsetup, steps, beta_high, gamma, ens=ensemble, monolothic_flag=True)

    #for gamma in [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10000.0]:
    #    twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=ensemble, monolothic_flag=False)
    #    twocell_ensemble_stats(simsetup, steps, beta, gamma, ens=ensemble, monolothic_flag=True)
