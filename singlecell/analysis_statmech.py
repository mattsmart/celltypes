import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.optimize import fsolve

from singlecell.singlecell_functions import label_to_state
from singlecell.singlecell_simsetup import singlecell_simsetup
from twocell.twocell_analysis import build_twocell_J_h, refine_applied_field_twocell


def params_unpack(params):
    return params['N'], params['beta'], params['r1'], params['r2'], params['kappa1'], params['kappa2']


def params_fill(params_base, pdict):
    params = params_base.copy()
    for k in list(pdict.keys()):
        params[k] = pdict[k]
    return params


def free_energy(m, params):
    N, beta, r1, r2, kappa1, kappa2 = params_unpack(params)
    term0 = (1 - r1 - r2) * np.log(np.cosh(beta * m))
    term1 = r1 * np.log(np.cosh(beta * (m - kappa1)))
    term2 = r2 * np.log(np.cosh(beta * (m + kappa2)))
    return N * m ** 2 / 2 - N/beta * (term0 + term1 + term2)


def free_energy_dm(m, params):
    N, beta, r1, r2, kappa1, kappa2 = params_unpack(params)
    term0 = (1 - r1 - r2) * np.tanh(beta * m)
    term1 = r1 * np.tanh(beta * (m - kappa1))
    term2 = r2 * np.tanh(beta * (m + kappa2))
    return m - (term0 + term1 + term2)


def get_all_roots(params, tol=1e-6):
    mGrid = np.linspace(-1.1, 1.1, 100)
    unique_roots = []
    for mTrial in mGrid:
        solution, infodict, _, _ = fsolve(free_energy_dm, mTrial, args=params, full_output=True)
        mRoot = solution[0]
        append_flag = True
        for k, usol in enumerate(unique_roots):
            if np.abs(mRoot - usol) <= tol:  # only store unique roots from list of all roots
                append_flag = False
                break
        if append_flag:
            if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
                unique_roots.append(mRoot)
    # remove those which are not stable (keep minima)
    return unique_roots


def is_stable(mval, params, eps=1e-4):
    fval_l = free_energy(mval - eps, params)
    fval = free_energy(mval, params)
    fval_r = free_energy(mval + eps, params)
    return (fval < fval_l and fval < fval_r)


def get_stable_roots(params, tol=1e-6):
    unique_roots = get_all_roots(params, tol=tol)
    stable_roots = [mval for mval in unique_roots if is_stable(mval, params)]
    return stable_roots


def plot_f_and_df(params, num_pts=20):
    fig, axarr = plt.subplots(1, 2)
    mvals = np.linspace(-2, 2, num_pts)
    axarr[0].plot(mvals, [free_energy(m, params) for m in mvals])
    axarr[0].set_ylabel(r'$f(m)$')
    axarr[1].plot(mvals, [free_energy_dm(m, params) for m in mvals])
    axarr[1].set_ylabel(r'$df/dm$')
    for idx in range(2):
        axarr[idx].set_xlabel(r'$m$')
        axarr[idx].axvline(x=-1, color='k', linestyle='--')
        axarr[idx].axvline(x=1, color='k', linestyle='--')
        axarr[idx].axhline(y=0, color='k', linestyle='-')
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    return


def get_fp_data_2d(p1, p2, p1range, p2range, params_base):
    fp_data_2d = np.zeros((len(p1range), len(p2range)), dtype=int)
    for i, p1val in enumerate(p1range):
        for j, p2val in enumerate(p2range):
            params = params_fill(params_base, {p1: p1val, p2: p2val})
            fp_data_2d[i,j] = len(get_stable_roots(params, tol=1e-6))
    return fp_data_2d


def get_data_wrapper(fn_args_dict):
    return get_fp_data_2d(*fn_args_dict['args'], **fn_args_dict['kwargs'])


def get_data_parallel(p1, p2, p1range, p2range, params, num_proc=cpu_count()):
    fn_args_dict = [0] * num_proc
    print("NUM_PROCESSES:", num_proc)
    assert len(p1range) % num_proc == 0
    for i in range(num_proc):
        range_step = len(p1range) / num_proc
        p1range_reduced = p1range[i * range_step: (1 + i) * range_step]
        print("process:", i, "job size:", len(p1range_reduced), "x", len(p2range))
        fn_args_dict[i] = {'args': (p1, p2, p1range_reduced, p2range, params),
                           'kwargs': {}}
        print(i, p1, np.min(p1range_reduced), np.max(p1range_reduced))
    t0 = time.time()
    pool = Pool(num_proc)
    results = pool.map(get_data_wrapper, fn_args_dict)
    pool.close()
    pool.join()
    print("TIMER:", time.time() - t0)

    results_dim = np.shape(results[0])
    results_collected = np.zeros((results_dim[0] * num_proc, results_dim[1]))
    for i, result in enumerate(results):
        results_collected[i * results_dim[0]:(i + 1) * results_dim[0], :] = result
    return results_collected


def make_phase_diagram(fp_data_2d, p1, p2, p1range, p2range, params_base):
    fs = 16
    # MAKE COLORBAR: https://stackoverflow.com/questions/9707676/defining-a-discrete-colormap-for-imshow-in-matplotlib
    assert np.max(fp_data_2d) <= 4
    cmap = clr.ListedColormap(['lightsteelblue', 'lightgrey', 'thistle', 'moccasin'])
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]
    norm = clr.BoundaryNorm(bounds, cmap.N)

    # regularize values
    # plot image
    fig = plt.figure(figsize=(5,10))
    img = plt.imshow(fp_data_2d, cmap=cmap, interpolation="none", origin='lower', aspect='auto', norm=norm,
                     extent=[p2range[0], p2range[-1], p1range[0], p1range[-1]])
    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle='-')
    ax.set_xlabel(p2, fontsize=fs)
    ax.set_ylabel(p1, fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    # add cbar
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[i-0.5 for i in bounds])

    """
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.savefig(outdir + os.sep + 'fig_fpsum_data_2d_%s_%s_%s.pdf' % (param_1_name, param_2_name, figname_mod),
               bbox_inches='tight')
    """
    plt.show()
    return plt.gca()


def free_energy_pdim(c, simsetup, beta=10**3):
    xi = simsetup['XI']
    xi_dot_c = np.dot(xi, c)  # this is N x 1 object
    term1 = 0.5 * np.dot(xi_dot_c, xi_dot_c)
    term2 = 0
    for idx in range(simsetup['N']):
        term2 += np.log( np.cosh( beta * xi_dot_c[idx] ) )
    term2_scaled = term2 / beta
    return term1 - term2_scaled


def free_energy_pdim_neg_grad(c, simsetup, beta=10**3):
    xi = simsetup['XI']
    xi_dot_c = np.dot(xi, c)  # this is N x 1 object
    tanh_factor = np.tanh(beta * xi_dot_c)
    cdot_term1 = np.dot(xi.T, xi_dot_c)
    cdot_term2 = np.zeros(simsetup['P'])
    for idx in range(simsetup['N']):
        for mu in range(simsetup['P']):
            cdot_term2[mu] += xi[idx, mu] * tanh_factor[idx]
    return -1 * cdot_term1 + cdot_term2


def free_energy_pdim_hessian(c, simsetup, beta=10**3):
    xi = simsetup['XI']
    xi_dot_c = np.dot(xi, c)  # this is N x 1 object
    sech_factor = 1/np.cosh(beta * xi_dot_c)
    A_unscaled = np.dot(xi.T, xi)
    hess_term1 = A_unscaled  # this is the first term of the p x p matrix
    hess_term2_unscaled = np.zeros((simsetup['P'], simsetup['P']))
    for mu in range(simsetup['P']):
        for nu in range(simsetup['P']):
            for idx in range(simsetup['N']):
                hess_term2_unscaled[mu, nu] += xi[idx, mu] * xi[idx, nu] * sech_factor[idx]
    hess = hess_term1 - hess_term2_unscaled * beta
    return hess


def unique_roots_check(unique_roots, solution, infodict, tol=1e-4):
    append_flag = True
    for k, usol in enumerate(unique_roots):
        if np.linalg.norm(solution - usol) <= tol:  # only store unique roots from list of all roots
            append_flag = False
            break
    if append_flag:
        if np.linalg.norm(infodict["fvec"]) <= 10e-3:  # only append actual roots (i.e. f(x)=0)
            unique_roots.append(solution)
            print(solution, np.linalg.norm(infodict["fvec"]), infodict["fvec"])
    return unique_roots


def pdim_fixedpoints_gridsearch(simsetup):
    c0_coord_step = 0.4
    c0_coord_init = -1 - 0.5 * c0_coord_step
    c0_base = c0_coord_init * np.ones(simsetup['P'])
    pts_per_axis = 1 + int((np.abs(c0_coord_init) - c0_coord_init) / c0_coord_step)
    num_pts = pts_per_axis ** simsetup['P']
    print("Running: pdim minima search with num_pts", num_pts, "step size", c0_coord_step)

    def step_vec(pt):
        step_vec = np.zeros(simsetup['P'], dtype=int)
        for mu in range(simsetup['P']):
            step_vec[mu] = (pt / (pts_per_axis ** mu)) % pts_per_axis
        return step_vec

    unique_roots = []
    # TODO parallelize
    for pt in range(num_pts):
        c0_pt = c0_base + c0_coord_step * step_vec(pt)
        solution, infodict, _, _ = fsolve(free_energy_pdim_neg_grad, c0_pt, args=simsetup, full_output=True)
        unique_roots = unique_roots_check(unique_roots, solution, infodict)

    return unique_roots


def pdim_fixedpoints_randomsearch(simsetup, num_pts=500):
    pdim_hypercube_corner = 1.05
    pdim_hypercube_edge = 2 * pdim_hypercube_corner
    c0_bottom_left = -pdim_hypercube_corner * np.ones(simsetup["P"])

    def c0_randomize():
        rvec = np.random.uniform(size=simsetup["P"])
        return c0_bottom_left + rvec * pdim_hypercube_edge

    unique_roots = []
    # TODO parallelize
    c0_pt_arr = np.zeros((num_pts,3))
    for pt in range(num_pts):
        c0_pt = c0_randomize()

        c0_pt_arr[pt, :] = c0_pt
        print(pt, c0_pt)

        solution, infodict, _, _ = fsolve(free_energy_pdim_neg_grad, c0_pt, args=simsetup, full_output=True)
        unique_roots = unique_roots_check(unique_roots, solution, infodict)

    plt.scatter(c0_pt_arr[:,0], c0_pt_arr[:,1], c0_pt_arr[:,2])
    return unique_roots


def minima_from_fixed_points(fixed_points, simsetup, beta=10**3, verbose=False):

    def check_if_minimum(c0, simsetup):
        hess = free_energy_pdim_hessian(c0, simsetup, beta=beta)
        eigenvalues, V = np.linalg.eig(hess)
        print("\n", "Hessian evals", eigenvalues)
        return all(np.real(eig) > 0 for eig in eigenvalues)

    minima = []
    for cRoot in fixed_points:
        boolv = check_if_minimum(cRoot, simsetup)
        print("gradient", free_energy_pdim_neg_grad(cRoot, simsetup))
        print("is minimum:", cRoot, boolv)
        if boolv:
            minima.append(cRoot)

    return minima


if __name__ == '__main__':
    simple_test = False
    phase_diagram = False
    pdim = False
    run_twocell = True

    if simple_test:
        params = {
            'N': 100.0,
            'beta': 100.0,
            'N1': 0,
            'N2': 0,
            'kappa1': 0.0,
            'kappa2': 0.0}
        print(get_all_roots(params, tol=1e-6))
        print(get_stable_roots(params, tol=1e-6))
        plot_f_and_df(params)

    if phase_diagram:
        params = {
            'N': 1000.0,
            'beta': 100.0,
            'r1': 0,
            'r2': 0,
            'kappa1': 0.0,
            'kappa2': 0.0}
        p1 = 'kappa1'
        p1range = np.linspace(0.1, 2.0, 12)
        p2 = 'r1'
        p2range = np.linspace(0.0, 0.50, 51)
        # parallelize scan
        fp_data_2d = get_data_parallel(p1, p2, p1range, p2range, params, num_proc=cpu_count())
        # plot data
        make_phase_diagram(fp_data_2d, p1, p2, p1range, p2range, params)

    if pdim:
        random_mem = False
        random_W = False
        # simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA, housekeeping=HOUSEKEEPING)
        simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, housekeeping=0, curated=True)
        print('note: N =', simsetup['N'], 'P =', simsetup['P'])
        #fixed_points = pdim_fixedpoints_gridsearch(simsetup)
        fixed_points = pdim_fixedpoints_randomsearch(simsetup, num_pts=500)
        minima = minima_from_fixed_points(fixed_points, simsetup)
        for idx, minimum in enumerate(minima):
            print(idx, minimum)

    if run_twocell:
        GAMMA = 1.0
        HOUSEKEEPING = 0.0
        KAPPA = 0
        manual_field = None
        FLAG_01 = False

        random_mem = False
        random_W = False
        # simsetup = singlecell_simsetup(unfolding=False, random_mem=random_mem, random_W=random_W, npzpath=MEMS_MEHTA, housekeeping=HOUSEKEEPING)
        simsetup = singlecell_simsetup(unfolding=True, random_mem=random_mem, random_W=random_W, housekeeping=0, curated=True)
        print('note: single cell N =', simsetup['N'], 'P =', simsetup['P'])
        N_multicell = simsetup['N'] * 2
        print('note: multi cell N =', N_multicell)

        statespace_multicell = np.array([label_to_state(label, N_multicell) for label in range(2 ** N_multicell)])
        J_multicell, h_multicell = build_twocell_J_h(simsetup, GAMMA, flag_01=FLAG_01)
        h_multicell = refine_applied_field_twocell(N_multicell, h_multicell, housekeeping=HOUSEKEEPING, kappa=KAPPA,
                                                   manual_field=manual_field)

        # TODO 2N spin MF minimization / root finding
        #fixed_points = pdim_fixedpoints_gridsearch(simsetup)
        fixed_points = pdim_fixedpoints_randomsearch(simsetup, num_pts=500)
        minima = minima_from_fixed_points(fixed_points, simsetup)
        for idx, minimum in enumerate(minima):
            print(idx, minimum)
