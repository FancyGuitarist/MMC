import unittest
from laminate import Laminate
from composite import CompositeType
import numpy as np


class TestLaminate(unittest.TestCase):
    def test_q_k(self):
        lam = Laminate(thetas=[0, 45, 0], h=0.005, composite_type=CompositeType.Graphite_Epoxy)
        q_k = lam.q_k/1e9
        expected_q_k = np.array([[359.37975721, 45.1147341, 35.89735324],
                                 [45.1147341, 72.20093126, 35.89735324],
                                 [35.89735324, 35.89735324, 49.26890253]])
        compare = np.round(q_k, 2) == np.round(expected_q_k, 2)
        self.assertTrue(compare.all())

    def test_a_matrix(self):
        lam = Laminate(thetas=[30, -30, 0, 0, -30, 30], h=0.150, composite_type=CompositeType.Graphite_Epoxy)
        a_matrix = lam.a_matrix / 1e6
        expected_a_matrix = np.array([[102.4036, 18.9448, 0], [18.9448, 16.2499, 0], [0, 0, 20.191]])
        compare = np.round(a_matrix, 2) == np.round(expected_a_matrix, 2)
        self.assertTrue(compare.all())

    def test_b_matrix(self):
        lam = Laminate(thetas=[30, -30, 0], h=0.150, composite_type=CompositeType.Graphite_Epoxy)
        b_matrix = lam.b_matrix
        expected_b_matrix = np.array([[1416.35, -608.66, -1050.89],
                                      [-608.66, -199.03, -348.07],
                                      [-1050.89, -348.07, -608.66]])
        compare = np.round(b_matrix, 2) == np.round(expected_b_matrix, 2)
        self.assertTrue(compare.all())

    def test_d_matrix(self):
        lam = Laminate(thetas=[0, 90, 90, 0], h=0.150, composite_type=CompositeType.Graphite_Epoxy)
        d_matrix = lam.d_matrix
        expected_d_matrix = np.array([[2.4804, 0.0543, 0], [0.0543, 0.5419, 0], [0, 0, 0.0792]])
        compare = np.round(d_matrix, 4) == np.round(expected_d_matrix, 4)
        self.assertTrue(compare.all())

    def test_abd_matrix(self):
        lam = Laminate(thetas=[30, -30, 0], h=0.150, composite_type=CompositeType.Graphite_Epoxy)
        a_matrix, b_matrix, d_matrix = lam.abd_matrix
        a_matrix = a_matrix / 1e6
        expected_a_matrix = np.array([[51.2, 9.47, 0], [9.47, 8.12, 0], [0, 0, 10.1]])
        expected_b_matrix = np.array([[1416.35, -608.66, -1050.89],
                                      [-608.66, -199.03, -348.07],
                                      [-1050.89, -348.07, -608.66]])
        expected_c_matrix = np.array([[0.93, 0.13, 0.16], [0.13, 0.13, 0.05], [0.16, 0.05, 0.14]])
        compare_a = np.round(a_matrix, 2) == np.round(expected_a_matrix, 2)
        compare_b = np.round(b_matrix, 2) == np.round(expected_b_matrix, 2)
        compare_c = np.round(d_matrix, 2) == np.round(expected_c_matrix, 2)
        self.assertTrue(compare_a.all())
        self.assertTrue(compare_b.all())
        self.assertTrue(compare_c.all())

# [0, 90, 90, 0]
# [30, -30, 0, 0, -30, 30]
