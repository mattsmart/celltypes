import numpy as np
import unittest

from data_analysis import build_basin_states
from singlecell_functions import hamiltonian, hamming, label_to_state


class TestExpt(unittest.TestCase):

    def test_something(self):
        print("RUNNING test_something")
        self.assertEqual(True, False)

    def test_build_basin_states_N3(self):
        print("RUNNING test_build_basin_states")

        intxn_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        memory_vec = np.array([1, 1, 1])
        flip1 = np.array([-1, 1, 1])
        flip2 = np.array([1, -1, 1])
        flip3 = np.array([1, 1, -1])
        flip12 = np.array([-1, -1, 1])
        flip23 = np.array([1, -1, -1])
        flip31 = np.array([-1, 1, -1])
        flip123 = np.array([-1, -1, -1])

        """
        states = [memory_vec, flip1, flip2, flip3, flip12, flip23, flip31, flip123]
        for state in states:
            print state
            print hamming(memory_vec, state)
            print hamiltonian(state, intxn_matrix=intxn_matrix)
        """
        basin = build_basin_states(intxn_matrix, memory_vec)
        print(basin)

        self.assertIn(tuple(memory_vec), basin[0])
        self.assertIn(tuple(flip1), basin[1])
        self.assertIn(tuple(flip2), basin[1])
        self.assertIn(tuple(flip3), basin[1])
        self.assertNotIn(tuple(flip12), basin[2])
        self.assertNotIn(tuple(flip23), basin[2])
        self.assertNotIn(tuple(flip31), basin[2])
        self.assertNotIn(tuple(flip123), basin[3])

    def test_build_basin_states_N4(self):
        N = 4
        memory_vec = np.array([1 for i in range(N)])
        memory_vec_as_column = [[1] for i in range(N)]
        XI = memory_vec_as_column
        intxn_matrix = np.dot(XI, np.transpose(XI))
        np.fill_diagonal(intxn_matrix, 0)

        labels_to_states = {idx: label_to_state(idx, N) for idx in range(2 ** N)}
        for k,v in labels_to_states.items():
            print("label", "state", "hamming(memory_vec, v)", "energy")
            print(k, v, hamming(memory_vec, v), hamiltonian(v, intxn_matrix=intxn_matrix))

        basin = build_basin_states(intxn_matrix, memory_vec)
        print(basin)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
