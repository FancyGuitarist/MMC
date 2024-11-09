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
        abd_matrix = lam.abd_matrix
        a_matrix = abd_matrix[:3, :3]
        b_matrix = abd_matrix[:3, 3:]
        d_matrix = abd_matrix[3:, 3:]
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

    def test_inv_abd_matrix(self):
        lam = Laminate(thetas=[30, -30, 0], h=0.150, composite_type=CompositeType.Graphite_Epoxy)
        inv_a_matrix = lam.inv_a_matrix * 1e9
        expected_a = np.array([[41.4649806, -39.83330754, 12.19402388], [-39.83330754, 185.53318223, 22.26465167], [12.19402388, 22.26465167, 150.37129891]])
        compare_a = np.round(inv_a_matrix, 2) == np.round(expected_a, 2)
        inv_b_matrix = lam.inv_b_matrix * 1e6
        expected_b = np.array([[-164.61527041, 179.43487707, 383.86117397], [207.26476213, -71.828884, 52.51080262], [18.79887382, 195.42479121, 706.95480147]])
        compare_b = np.round(inv_b_matrix, 2) == np.round(expected_b, 2)
        inv_d_matrix = lam.inv_d_matrix
        expected_d = np.array([[2.09888038, -1.55292858, -2.42398039], [-1.55292858, 10.8394884, -0.27604197], [-2.42398039, -0.27604197, 16.06881313]])
        compare_d = np.round(inv_d_matrix, 2) == np.round(expected_d, 2)
        self.assertTrue(compare_a.all())
        self.assertTrue(compare_b.all())
        self.assertTrue(compare_d.all())
