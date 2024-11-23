import unittest
import math
from laminate import Laminate, Variables, LaminateAngles, FailureCriteria
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
        lam = Laminate(thetas=[30, -30, 0, 0, -30, 30], composite_type=CompositeType.Graphite_Epoxy)
        a_matrix = lam.a_matrix / 1e6
        expected_a_matrix = np.array([[102.4036, 18.9448, 0], [18.9448, 16.2499, 0], [0, 0, 20.191]])
        compare = np.round(a_matrix, 2) == np.round(expected_a_matrix, 2)
        self.assertTrue(compare.all())

    def test_b_matrix(self):
        lam = Laminate(thetas=[30, -30, 0], composite_type=CompositeType.Graphite_Epoxy)
        b_matrix = lam.b_matrix
        expected_b_matrix = np.array([[1416.35, -608.66, -1050.89],
                                      [-608.66, -199.03, -348.07],
                                      [-1050.89, -348.07, -608.66]])
        compare = np.round(b_matrix, 2) == np.round(expected_b_matrix, 2)
        self.assertTrue(compare.all())

    def test_d_matrix(self):
        lam = Laminate(thetas=[0, 90, 90, 0], composite_type=CompositeType.Graphite_Epoxy)
        d_matrix = lam.d_matrix
        expected_d_matrix = np.array([[2.4804, 0.0543, 0], [0.0543, 0.5419, 0], [0, 0, 0.0792]])
        compare = np.round(d_matrix, 4) == np.round(expected_d_matrix, 4)
        self.assertTrue(compare.all())

    def test_abd_matrix(self):
        lam = Laminate(thetas=[30, -30, 0], composite_type=CompositeType.Graphite_Epoxy)
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
        lam = Laminate(thetas=[30, -30, 0], composite_type=CompositeType.Graphite_Epoxy)
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

    def test_solver_ms(self):
        lam = Laminate(thetas=[0, 90, 90, 0], composite_type=CompositeType.Graphite_Epoxy)
        solved = lam.solve_eps_kap_n_m(epsilons=[0, 0, 0], kappas=[3.33, 0, 0])
        res = {key: round(solved[key], 4) for key in solved}
        expected = {Variables.M_x: 8.25967878053540, Variables.M_xy: 0.0, Variables.M_y: 0.180735714772624}
        expected = {key: round(solved[key], 4) for key in expected}
        self.assertEqual(res[Variables.M_x], expected[Variables.M_x])
        self.assertEqual(res[Variables.M_y], expected[Variables.M_y])

    def test_solver_ns(self):
        lam = Laminate(thetas=[0, 90, 90, 0], composite_type=CompositeType.Graphite_Epoxy)
        solved = lam.solve_eps_kap_n_m(epsilons=[1000/1e6, 0, 0], kappas=[0, 0, 0])
        res = {key: round(solved[key] / 1e3, 3) for key in solved}
        expected = {Variables.N_x: 50371.8493528608, Variables.N_xy: -6.30261859946559e-15, Variables.N_y: 1809.16631404027}
        expected = {key: round(solved[key] / 1e3, 3) for key in expected}
        self.assertEqual(res[Variables.N_x], expected[Variables.N_x])
        self.assertEqual(res[Variables.N_xy], expected[Variables.N_xy])
        self.assertEqual(res[Variables.N_y], expected[Variables.N_y])

    def test_solver_eps(self):
        lam = Laminate(thetas=[0, 0, 90, 90], composite_type=CompositeType.Graphite_Epoxy)
        solved = lam.solve_eps_kap_n_m(ns=[10**3, 0, 0], ms=[0, 0, 0])
        expected = {Variables.eps_x: 4.40952998181854e-5,
                    Variables.eps_y: -1.58373639370929e-6,
                    Variables.kap_x: 0.188546329862917}
        abs_tol = 1/1e2
        self.assertTrue(math.isclose(solved[Variables.eps_x], expected[Variables.eps_x] * 1e6, abs_tol=abs_tol))
        self.assertTrue(math.isclose(solved[Variables.eps_y], expected[Variables.eps_y] * 1e6, abs_tol=abs_tol))
        self.assertTrue(math.isclose(solved[Variables.kap_x], expected[Variables.kap_x], abs_tol=abs_tol))

    def test_laminate_thermal_coefficients(self):
        lam = Laminate(thetas=[0, 60], composite_type=CompositeType.Graphite_Epoxy, delta_t=150)
        thermal_coeffs = lam.laminate_expansion_coefficients[0]
        expected = np.array([46.445, 63.314, -14.609, 0.001898, -0.001898, -0.001096])
        compare_ns = np.round(thermal_coeffs[:3], 3) == np.round(expected[:3], 3)
        compare_ms = np.round(thermal_coeffs[3:], 5) == np.round(expected[3:], 5)
        self.assertTrue(compare_ns.all())
        self.assertTrue(compare_ms.all())

    def test_solver_w_delta_t(self):
        lam = Laminate(thetas=LaminateAngles("[±30]S"), composite_type=CompositeType.Graphite_Epoxy, delta_t=-150)
        solution = lam.solve_eps_kap_n_m(ns=[0, 0, 0], ms=[0, 0, 0])
        expected = {Variables.eps_x: 650.1, Variables.eps_y: -2638.6, Variables.gam_xy: 0.0}
        res = {key: round(solution[key]) for key in solution}
        expected = {key: round(expected[key]) for key in expected}
        self.assertEqual(res[Variables.eps_x], expected[Variables.eps_x])
        self.assertEqual(res[Variables.eps_y], expected[Variables.eps_y])
        self.assertEqual(res[Variables.gam_xy], expected[Variables.gam_xy])

    def test_solve_residual_stresses(self):
        lam = Laminate(thetas=LaminateAngles("[±30]S"), composite_type=CompositeType.Graphite_Epoxy, delta_t=-150)
        solution = lam.solve_eps_kap_n_m(ns=[0, 0, 0], ms=[0, 0, 0])
        residual_stresses = lam.solve_residual_stresses(solution)
        expected = {30: {Variables.sigma_x: 0.0, Variables.sigma_y: 0.0, Variables.tau_xy: -25.063},
                    -30: {Variables.sigma_x: 0.0, Variables.sigma_y: 0.0, Variables.tau_xy: 25.063}}
        abs_tol = 1 / 1e2
        for key in expected:
            for var in expected[key]:
                self.assertTrue(math.isclose(residual_stresses[key][var], expected[key][var], abs_tol=abs_tol))

    def test_local_residual_stresses(self):
        # Still need a demonstration to build unittest, for now just checking if function outputs something
        lam = Laminate(thetas=LaminateAngles("[±30]S"), composite_type=CompositeType.Graphite_Epoxy, delta_t=-150)
        solution = lam.solve_eps_kap_n_m(ns=[0, 0, 0], ms=[0, 0, 0])
        local_residual_stresses = lam.local_residual_stresses(solution)
        # print(local_residual_stresses)

    # @unittest.skip("No tests for plotting, only for demonstration")
    def test_plot_curvature(self):
        lam = Laminate(thetas=LaminateAngles("[0_2/ 90_2]"), composite_type=CompositeType.Graphite_Epoxy, delta_t=-150)
        solution = lam.solve_eps_kap_n_m(ns=[0, 0, 0], ms=[0, 0, 0])
        lam.plot_curvature(solution, plate_dimensions=(0.25, 0.25))

    def test_pressure_failure_tsai_hill(self):
        lam = Laminate(thetas=LaminateAngles("[±60_6]S"), composite_type=CompositeType.Graphite_Epoxy, delta_t=-150)
        pressures = lam.failure_pressure_criteria(d=0.6, criteria=FailureCriteria.TsaiHill)
        expected = {60: 1.81564673079054, -60: 1.81564673079054}
        self.assertTrue(math.isclose(pressures[60], expected[60], abs_tol=1/1e2))
        self.assertTrue(math.isclose(pressures[-60], expected[-60], abs_tol=1/1e2))


class TestLaminateAngles(unittest.TestCase):
    def test_plus_minus(self):
        angles = "[0/ ±45]"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45])

    def test_symmetry(self):
        angles = "[0/ 45/ 90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, 90, 90, 45, 0])

    def test_symmetry_w_plus_minus(self):
        angles = "[0/ ±45/ 90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45, 90, 90, -45, 45, 0])

    def test_non_symmetry(self):
        angles = "[0/ 45/ ¬90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, 90, 45, 0])

    def test_non_symmetry_w_plus_minus(self):
        angles = "[0/ ±45/ ¬90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45, 90, -45, 45, 0])

    def test_repeat(self):
        angles = "[0/ 45_3/ 90]"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, 45, 45, 90])

    def test_repeat_w_plus_minus(self):
        angles = "[0/ ±45_3/ 90]"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45, 45, -45, 45, -45, 90])

    def test_repeat_w_symmetry(self):
        angles = "[0/ 45_3/ 90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, 45, 45, 90, 90, 45, 45, 45, 0])

    def test_repeat_w_symmetry_and_plus_minus(self):
        angles = "[0/ ±45_2/ 90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45, 45, -45, 90, 90, -45, 45, -45, 45, 0])

    def test_repeat_w_non_symmetry_and_plus_minus(self):
        angles = "[0/ ±45_2/ ¬90]S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, -45, 45, -45, 90, -45, 45, -45, 45, 0])

    def test_multiple_symmetries(self):
        angles = "[0/ 45/ 90]2S"
        lam_angles = LaminateAngles(angles)
        self.assertEqual(lam_angles.get_angles_list(), [0, 45, 90, 0, 45, 90, 90, 45, 0, 90, 45, 0])

    def test_multiple_symmetries_w_non_symmetry(self):
        angles = "[0/ 45/ ¬90]2S"
        lam_angles = LaminateAngles(angles)
        self.assertRaises(ValueError, lam_angles.get_angles_list)