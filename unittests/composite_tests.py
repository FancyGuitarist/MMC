import unittest
from composite import Composite, CompositeType, ExpansionType
from sympy import symbols, Matrix, linsolve


class TestComposite(unittest.TestCase):
    def setUp(self):
        self.sigma_x, self.sigma_y, self.tau_xy = symbols('sigma_x sigma_y tau_xy')
        self.epsilon_x, self.epsilon_y, self.gamma_xy = symbols('epsilon_x epsilon_y gamma_xy')

    def test_update_variables(self):
        composite = Composite(composite_type=CompositeType.Graphite_Epoxy)
        variables = composite.update_variables((None, 10, None), stress=True)
        self.assertEqual(composite.variables_to_solve, [self.sigma_x, self.tau_xy])
        self.assertEqual(variables, [self.sigma_x, 10, self.tau_xy])

    def test_reset_variables(self):
        composite = Composite(composite_type=CompositeType.Graphite_Epoxy)
        composite.update_variables((None, 10, None), stress=True)
        composite.reset_variables_to_solve()
        self.assertEqual(composite.variables_to_solve, [])

    def test_composite_w_angle(self):
        sigma_x, sigma_y, tau_xy = self.sigma_x, self.sigma_y, self.tau_xy
        composite = Composite(angle=30, composite_type=CompositeType.Graphite_Epoxy)
        actual = composite.solve_strains_and_stresses(strains=(1000, 0, 0), stresses=(None, None, None))
        actual = {key: round(float(value), 1) for key, value in actual.items()}
        expected = {sigma_x: 92.8, sigma_y: 30.1, tau_xy: 46.7}
        self.assertEqual(actual[sigma_x], expected[sigma_x])
        self.assertEqual(actual[sigma_y], expected[sigma_y])
        self.assertEqual(actual[tau_xy], expected[tau_xy])

    def test_composite_w_delta_t(self):
        sigma_x, sigma_y, tau_xy = self.sigma_x, self.sigma_y, self.tau_xy
        composite = Composite(delta_t=50, composite_type=CompositeType.Graphite_Epoxy)
        actual = composite.solve_strains_and_stresses(strains=(0, 0, 0), stresses=(None, None, None))
        actual = {key: round(float(value), 2) for key, value in actual.items()}
        expected = {sigma_x: -3.52, sigma_y: -14.77, tau_xy: 0.00}
        self.assertEqual(actual[sigma_x], expected[sigma_x])
        self.assertEqual(actual[sigma_y], expected[sigma_y])
        self.assertEqual(actual[tau_xy], expected[tau_xy])

    def test_composite_w_deltas(self):
        epsilon_x, epsilon_y, gamma_xy = self.epsilon_x, self.epsilon_y, self.gamma_xy
        composite = Composite(delta_t=-100, delta_m=0.2, composite_type=CompositeType.Graphite_Epoxy)
        actual = composite.solve_strains_and_stresses(strains=(None, None, None), stresses=(0, 0, 0))
        actual = {key: round(float(value), 0) for key, value in actual.items()}
        expected = {epsilon_x: 31, epsilon_y: -1476, gamma_xy: 0}
        self.assertEqual(actual[epsilon_x], expected[epsilon_x])
        self.assertEqual(actual[epsilon_y], expected[epsilon_y])
        self.assertEqual(actual[gamma_xy], expected[gamma_xy])