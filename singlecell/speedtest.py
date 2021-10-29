import os
# if intelypython
#os.environ['MKL_NUM_THREADS'] = '1'
#os.environ["MKL_THREADING_LAYER"] = "sequential"

# if openblas
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_MAIN_FREE"] = "1"                         # nope
#os.environ["OMP_NUM_THREADS"] = "1"

# unsure
#os.environ["NUMEXPR_NUM_THREADS"] = "1"


import time
import numpy as np
from multiprocessing import Pool


def foo_inner(A, b, n):
    for i in range(n):
        prod = np.dot(A, b)
    return prod


def worker_wrapper(fn_args):
    return foo_inner(*fn_args)


def foo_multiproc(A, b, num_processes, ensemble):
    # prepare fn args and kwargs for wrapper
    kwargs_dict = {}
    fn_args_dict = [0] * num_processes
    print(len(fn_args_dict), num_processes)
    assert ensemble % num_processes == 0
    for i in range(num_processes):
        subensemble = ensemble / num_processes
        fn_args_dict[i] = (A, b, subensemble)
    print(len(fn_args_dict))

    # generate results list over workers
    print("pooling")
    pool = Pool(num_processes)
    results = pool.map(worker_wrapper, fn_args_dict)
    pool.close()
    pool.join()
    print("done")

    # collect pooled results
    summed_results = np.zeros(1000)
    for i, result in enumerate(results):
        summed_results += results
    return results


if __name__ == '__main__':
    # settings
    parallel = False
    N = int(1e4)
    num_processes = 1

    # simulate
    t0 = time.time()
    A = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1)
    if parallel:
        foo_multiproc(A, b, num_processes, N)
    else:
        foo_inner(A, b, N)
    t1 = time.time()

    # conclude
    print("TIME:", t1-t0)
