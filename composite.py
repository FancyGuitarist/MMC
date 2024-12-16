import numpy as np
from enum import StrEnum

from sympy import symbols, Eq, solve
import sympy
from sympy.matrices import Matrix


class Variables:
    # For Composites:
    epsilon_x, epsilon_y, gamma_xy = symbols("epsilon_x, epsilon_y, gamma_xy")
    epsilon_1, epsilon_2, gamma_12 = symbols("epsilon_1, epsilon_2, gamma_12")
    sigma_x, sigma_y, tau_xy = symbols("sigma_x, sigma_y, tau_xy")
    sigma_1, sigma_2, tau_12 = symbols("sigma_1, sigma_2, tau_12")
    default_strains = [epsilon_x, epsilon_y, gamma_xy]
    default_local_strains = [epsilon_1, epsilon_2, gamma_12]
    default_stresses = [sigma_x, sigma_y, tau_xy]
    default_local_stresses = [sigma_1, sigma_2, tau_12]
    # For Laminates:
    eps_x, eps_y, gam_xy = symbols("eps_x, eps_y, gam_xy")
    kap_x, kap_y, kap_xy = symbols("kap_x, kap_y, kap_xy")
    N_x, N_y, N_xy = symbols("N_x, N_y, N_xy")
    M_x, M_y, M_xy = symbols("M_x, M_y, M_xy")
    default_eps = [eps_x, eps_y, gam_xy]
    default_kap = [kap_x, kap_y, kap_xy]
    default_N = [N_x, N_y, N_xy]
    default_M = [M_x, M_y, M_xy]


class ExpansionType(StrEnum):
    """
    Expansion types for expansion coefficients
    """
    Thermal = "Thermal"
    Hygroscopic = "Hygroscopic"


class CompositeType(StrEnum):
    """
    Enum for Composite Types.
    Can return the reduced Q and S matrices for the given Composite Type.
    """
    Glass_Epoxy = "Glass Epoxy"
    Graphite_Epoxy = "Graphite Epoxy"
    Lab_Composite = "Lab Composite"

    @property
    def q_3x3_matrix(self) -> np.ndarray:
        Q11 = self.properties['Q11']
        Q22 = self.properties['Q22']
        Q12 = self.properties['Q12']
        Q66 = self.properties['Q66']
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    @property
    def s_3x3_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.q_3x3_matrix)

    @property
    def properties(self) -> dict:
        """
        Return the properties of the Composite Type as a dictionary.
        :return:
        """
        if self == CompositeType.Glass_Epoxy:
            properties_dico = {'E1': 50 * 1e9, 'E2': 15.2 * 1e9, 'nu_12': 0.254, 'Q66': 4.70 * 1e9,
                               'nu_13': 0.254, 'nu_23': 0.428}
        elif self == CompositeType.Graphite_Epoxy:
            properties_dico = {'E1': 155 * 1e9, 'E2': 12.1 * 1e9, 'nu_12': 0.248, 'Q66': 4.40 * 1e9,
                               'nu_13': 0.248, 'nu_23': 0.458}
        elif self == CompositeType.Lab_Composite:
            properties_dico = {'E1': 114 * 1e9, 'E2': 5.4 * 1e9, 'nu_12': 0.3, 'Q66': 1.37 * 1e9,
                               'nu_13': 0.3, 'nu_23': 0.3}
        else:
            raise ValueError("Invalid Composite Type")
        nu_21 = (properties_dico['E2'] / properties_dico['E1']) * properties_dico['nu_12']
        Q11 = properties_dico['E1'] / (1 - properties_dico['nu_12'] * nu_21)
        Q22 = properties_dico['E2'] / (1 - properties_dico['nu_12'] * nu_21)
        Q12 = properties_dico['nu_12'] * properties_dico['E2'] / (1 - properties_dico['nu_12'] * nu_21)
        new_values = {'Q11': Q11, 'Q22': Q22, 'Q12': Q12, 'nu_21': nu_21}
        properties_dico.update(new_values)
        return properties_dico

    @property
    def safety_properties(self) -> dict:
        """
        Return the safety properties of the Composite Type as a dictionary.
        :return:
        """
        if self == CompositeType.Glass_Epoxy:
            safety_properties = {'sigma_1t': 1000 * 1e6, 'sigma_1c': -600 * 1e6, 'sigma_2t': 30 * 1e6,
                                 'sigma_2c': -120 * 1e6, 'tau_12f': 70 * 1e6}
        elif self == CompositeType.Graphite_Epoxy:
            safety_properties = {'sigma_1t': 1500 * 1e6, 'sigma_1c': -1250 * 1e6, 'sigma_2t': 50 * 1e6,
                                 'sigma_2c': -200 * 1e6, 'tau_12f': 100 * 1e6}
        else:
            raise ValueError("Invalid Composite Type")
        return safety_properties

    def expansion_coeffs(self, expansion_type: ExpansionType) -> dict:
        """
        Return the expansion coefficients for the given Composite Type and Expansion Type.
        :param expansion_type:
        :return:
        """
        if self == CompositeType.Graphite_Epoxy:
            dico = {ExpansionType.Thermal: {'alpha_1': -0.018 * 1e-6, 'alpha_2': 24.3 * 1e-6,
                                            'alpha_3': 24.3 * 1e-6},
                    ExpansionType.Hygroscopic: {'beta_1': 146 * 1e-6, 'beta_2': 4770 * 1e-6,
                                                'beta_3': 4770 * 1e-6}}
        elif self == CompositeType.Glass_Epoxy:
            dico = {ExpansionType.Thermal: {'alpha_1': 6.34 * 1e-6, 'alpha_2': 23.3 * 1e-6,
                                            'alpha_3': 23.3 * 1e-6},
                    ExpansionType.Hygroscopic: {'beta_1': 434 * 1e-6, 'beta_2': 6320 * 1e-6,
                                                'beta_3': 6320 * 1e-6}}
        else:
            raise ValueError("Invalid Composite Type")
        return dico[expansion_type]


class Composite:
    """
    Composite Class to solve_strains_and_stresses for the undefined variables in a Composite Material.
    Accepts the composite type, the angle of the fibers, the temperature delta and the moisture delta.
    """

    def __init__(self, composite_type: CompositeType, angle: int | float = 0, delta_t: int | float = 0,
                 delta_m: int | float = 0):
        """
        Initialize the Composite Class.
        Only required argument is the Composite Type.
        Fiber angle, temperature delta and humidity delta are optional arguments.
        :param composite_type:
        :param angle: Angle in degrees
        :param delta_t: Temperature delta in Celsius
        :param delta_m: Humidity delta in percentage
        """
        self.angle = np.radians(angle)
        self.composite_type = composite_type
        self.variables = Variables.default_strains + Variables.default_stresses
        self.variables_to_solve = []
        self.delta_t = delta_t
        self.delta_m = delta_m

    def reset_variables_to_solve(self):
        self.variables_to_solve = []

    @property
    def t_matrix(self) -> np.ndarray:
        """
        Transformation Matrix for angled fibers. Uses the angle in radians.
        """
        m, n = np.cos(self.angle), np.sin(self.angle)
        return np.array([[m ** 2, n ** 2, 2 * m * n], [n ** 2, m ** 2, -2 * m * n], [-m * n, m * n, m ** 2 - n ** 2]])

    @property
    def t_inv_matrix(self) -> np.ndarray:
        """
        Inverse of the Transformation Matrix.
        """
        return np.linalg.inv(self.t_matrix)

    @property
    def t_transposed_matrix(self) -> np.ndarray:
        """
        Transposed Transformation Matrix.
        """
        return np.transpose(self.t_matrix)

    @property
    def global_s_matrix(self) -> np.ndarray:
        """
        S Matrix in the global referential.
        """
        return self.t_transposed_matrix @ self.composite_type.s_3x3_matrix @ self.t_matrix

    @property
    def global_q_matrix(self) -> np.ndarray:
        """
        Q Matrix in the global referential.
        """
        return np.linalg.inv(self.global_s_matrix)

    @property
    def r_matrix(self) -> np.ndarray:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])

    @property
    def global_thermal_coeffs(self) -> np.ndarray:
        """
        Thermal coefficients in the global referential.
        :return: (alpha_x, alpha_y, alpha_xy, alpha_z)
        """
        thermal_coeffs = self.composite_type.expansion_coeffs(ExpansionType.Thermal)
        alpha_1, alpha_2, alpha_3 = thermal_coeffs['alpha_1'], thermal_coeffs['alpha_2'], thermal_coeffs['alpha_3']
        return self.local_to_global_coeffs((alpha_1, alpha_2, alpha_3))

    @property
    def global_hygroscopic_coeffs(self) -> np.ndarray:
        """
        Hygroscopic coefficients in the global referential.
        :return: (beta_x, beta_y, beta_xy, beta_z)
        """
        hygroscopic_coeffs = self.composite_type.expansion_coeffs(ExpansionType.Hygroscopic)
        beta_1, beta_2, beta_3 = hygroscopic_coeffs['beta_1'], hygroscopic_coeffs['beta_2'], hygroscopic_coeffs[
            'beta_3']
        return self.local_to_global_coeffs((beta_1, beta_2, beta_3))

    def local_to_global_coeffs(self, values: tuple) -> np.ndarray:
        """
        Transform the local coefficients to the global referential.
        :param values: (alpha_1, alpha_2, alpha_3) or (beta_1, beta_2, beta_3)
        :return: (alpha_x, alpha_y, alpha_xy, alpha_z) or (beta_x, beta_y, beta_xy, beta_z)
        """
        local_1, local_2, local_3 = values
        local_x = local_1 * np.cos(self.angle) ** 2 + local_2 * np.sin(self.angle) ** 2
        local_y = local_1 * np.sin(self.angle) ** 2 + local_2 * np.cos(self.angle) ** 2
        local_xy = 2 * (local_1 - local_2) * np.sin(self.angle) * np.cos(self.angle)
        local_z = local_3
        return np.array([local_x, local_y, local_xy, local_z])

    def update_variables(self, values: tuple, stress: bool = True) -> list:
        """
        Update the variables to solve_strains_and_stresses for the undefined variables.
        If a value is given, remove the corresponding variable from the list of variables to solve_strains_and_stresses.
        If value is None, returns the symbol for sympy to solve_strains_and_stresses.
        :param values: (sigma_x, sigma_y, tau_xy) or (epsilon_x, epsilon_y, gamma_xy)
        :param stress: True if the values are stresses, False if they are strains
        :return: Updated list of values with symbols or values
        """
        updated_values = []
        for index, value in enumerate(values):
            if stress:
                index = index + 3
            if value is None:
                value = self.variables[index]
                self.variables_to_solve.append(value)
            updated_values.append(value)
        return updated_values

    def stress_matrix(self, values: tuple) -> Matrix:
        """
        Create the stress matrix from the given values.
        :param values: (sigma_x, sigma_y, tau_xy)
        :return:
        """
        variables = self.update_variables(values, stress=True)
        matrix = Matrix([[variables[0]], [variables[1]], [variables[2]]]) * 1e6
        return matrix

    def strain_matrix(self, values: tuple) -> Matrix:
        """
        Create the strain matrix from the given values.
        :param values: (epsilon_x, epsilon_y, gamma_xy)
        :return:
        """
        variables = self.update_variables(values, stress=False)
        total_matrix = None
        for index, variable in enumerate(variables):
            matrix = Matrix(
                [variable * 1e-6 - (self.global_thermal_coeffs[index] * self.delta_t) - (
                        self.global_hygroscopic_coeffs[index] * self.delta_m)])
            if index == 0:
                total_matrix = matrix
            else:
                total_matrix = total_matrix.row_insert(index, matrix)
        return total_matrix

    def solve_strains_and_stresses(self, strains: tuple, stresses: tuple) -> dict:
        """
        Solve for the Composite's undefined variables
        :param strains: epsilon_x (micro_epsilon), epsilon_y (micro_epsilon), gamma_xy (micro_rad)
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: Dictionary of the solved variables
        """
        self.reset_variables_to_solve()
        strain_matrix = self.strain_matrix(strains)
        stress_matrix = self.stress_matrix(stresses)
        equation = Eq(Matrix(self.global_s_matrix) * stress_matrix, strain_matrix)
        solution = solve(equation, self.variables_to_solve)
        return solution

    def epsilon_3(self, stresses: tuple):
        """
        Calculate the third strain component from the given stresses.
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: epsilon_3 (micro_epsilon)
        """
        properties = self.composite_type.properties
        S_13 = -properties['nu_13'] / properties['E1']
        S_23 = -properties['nu_23'] / properties['E2']
        return S_13 * (stresses[0] * 1e6) + S_23 * (stresses[1] * 1e6)

    def epsilon_z(self, stresses: tuple):
        """
        Calculate the third strain component from the given stresses.
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: epsilon_3 (micro_epsilon)
        """
        beta_3 = self.composite_type.expansion_coeffs(ExpansionType.Hygroscopic)['beta_3']
        alpha_3 = self.composite_type.expansion_coeffs(ExpansionType.Thermal)['alpha_3']
        properties = self.composite_type.properties
        S_13 = -properties['nu_13'] / properties['E1']
        S_23 = -properties['nu_23'] / properties['E2']
        epsilon_z = (((alpha_3 * self.delta_t + beta_3 * self.delta_m
                       + (S_13 * np.cos(self.angle) ** 2 + S_23 * np.sin(self.angle) ** 2) * (stresses[0] * 1e6))
                      + (S_13 * np.sin(self.angle) ** 2 + S_23 * np.cos(self.angle) ** 2) * (stresses[1] * 1e6))
                     + 2 * (S_13 - S_23) * np.sin(self.angle) * np.cos(self.angle) * (stresses[2] * 1e6))
        return epsilon_z

    def solve_radial_stresses(self, pressure: float, diameter: float, thickness: float):
        """
        Solve for the stresses in a cylindrical composite.
        :param pressure: Inside pressure (MPa)
        :param diameter: Diameter of the cylinder (m)
        :param thickness: Thickness of the cylinder (mm)
        :return: strains: sigma_a (MPa), sigma_h (MPa)
        """
        r = diameter / 2
        thickness = thickness / 1000
        pressure = pressure * 1e6
        sigma_a = (pressure * r) / (thickness * 2)
        sigma_h = pressure * r / thickness
        return {'sigma_a': sigma_a / 1e6, 'sigma_h': sigma_h / 1e6}

    def mechanical_strains(self, values: tuple = (Variables.epsilon_x, Variables.epsilon_y, Variables.gamma_xy)):
        mec_strains = []
        for index, strain in enumerate(values):
            mec_strains.append(strain * 1e-6 - (self.global_thermal_coeffs[index] * self.delta_t) - (
                    self.global_hygroscopic_coeffs[index] * self.delta_m))
        return np.array(mec_strains)

    def global_to_local_strains(self, values: tuple):
        return self.r_matrix @ self.t_matrix @ np.linalg.inv(self.r_matrix) @ values

    def global_to_local_stresses(self, values: tuple):
        return self.t_matrix @ values

    def f_ij_elements(self):
        properties = self.composite_type.safety_properties
        sigma_1t, sigma_1c = properties['sigma_1t'], properties['sigma_1c']
        sigma_2t, sigma_2c, tau_12f = properties['sigma_2t'], properties['sigma_2c'], properties['tau_12f']
        F1 = (1 / sigma_1t) + (1 / sigma_1c)
        F11 = -1 / (sigma_1t * sigma_1c)
        F66 = (1 / tau_12f) ** 2
        F2 = (1 / sigma_2t) + (1 / sigma_2c)
        F22 = -1 / (sigma_2t * sigma_2c)
        F12 = -0.5 * np.sqrt(F11 * F22)
        dico = {'F1': F1, 'F11': F11, 'F66': F66, 'F2': F2, 'F22': F22, 'F12': F12}
        return dico

    def fs_max(self, values: tuple):
        """
        Calculate the maximum safety factor for the given stresses.
        :param values: sigma_1 (MPa), sigma_2 (MPa), tau_12 (MPa)
        :return: Fs_max
        """
        properties = self.composite_type.safety_properties
        sigma_1, sigma_2, tau_12 = values[0] * 1e6, values[1] * 1e6, values[2] * 1e6
        security_factors = []
        if sigma_1 == 0:
            security_factors.append(float('inf'))
        elif sigma_1 > 0:
            security_factors.append(properties['sigma_1t'] / sigma_1)
        else:
            security_factors.append(properties['sigma_1c'] / sigma_1)
        if sigma_2 == 0:
            security_factors.append(float('inf'))
        elif sigma_2 > 0:
            security_factors.append(properties['sigma_2t'] / sigma_2)
        else:
            security_factors.append(properties['sigma_2c'] / sigma_2)
        if tau_12 == 0:
            security_factors.append(float('inf'))
        elif tau_12 > 0:
            security_factors.append(properties['tau_12f'] / tau_12)
        else:
            security_factors.append(-properties['tau_12f'] / tau_12)
        return min(security_factors)

    def fs_tsai_wu(self, values: tuple):
        """
        Calculate the safety factor using the Tsai-Wu criterion.
        :param values: sigma_1 (MPa), sigma_2 (MPa), tau_12 (MPa)
        :return: Fs_tsai_wu
        """
        sigma_1, sigma_2, tau_12 = values[0] * 1e6, values[1] * 1e6, values[2] * 1e6
        Fs_tsai_wu = symbols('Fs_tsai_wu')
        f_ijs = self.f_ij_elements()
        a = (f_ijs['F11'] * sigma_1 ** 2 + f_ijs['F22'] * sigma_2 ** 2 + f_ijs['F66'] * tau_12 ** 2
             - sigma_1 * sigma_2 * np.sqrt(f_ijs['F11'] * f_ijs['F22']))
        b = f_ijs['F1'] * sigma_1 + f_ijs['F2'] * sigma_2
        eq2 = Eq(1, a * Fs_tsai_wu ** 2 + b * Fs_tsai_wu)
        return max(solve(eq2, Fs_tsai_wu))

    def fs_tsai_hill(self, values: tuple):
        """
        Calculate the safety factor using the Tsai-Hill criterion.
        :param values: sigma_1 (MPa), sigma_2 (MPa), tau_12 (MPa)
        :return: Fs_tsai_hill
        """
        properties = self.composite_type.safety_properties
        sigma_1, sigma_2, tau_12 = values[0] * 1e6, values[1] * 1e6, values[2] * 1e6
        sigma_1t, sigma_1c = properties['sigma_1t'], properties['sigma_1c']
        sigma_2t, sigma_2c, tau_12f = properties['sigma_2t'], properties['sigma_2c'], properties['tau_12f']
        sigma_1R = sigma_1t if sigma_1 > 0 else sigma_1c
        sigma_2R = sigma_2t if sigma_2 > 0 else sigma_2c
        Fs_tsai_hill = symbols('Fs_tsai_hill')
        eq1 = Eq(1, Fs_tsai_hill ** 2 * (
                (sigma_1 / sigma_1R) ** 2 + (sigma_2 / sigma_2R) ** 2 - (sigma_1 * sigma_2 / sigma_1R ** 2) + (
                tau_12 / tau_12f) ** 2))
        return max(solve(eq1, Fs_tsai_hill))


if __name__ == '__main__':
    from unittests.composite_tests import *
    unittest.main()
