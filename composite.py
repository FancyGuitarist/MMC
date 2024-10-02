import numpy as np
from enum import StrEnum

from sympy import symbols, Eq, solve
from sympy.matrices import Matrix

epsilon_x, epsilon_y, gamma_xy = symbols("epsilon_x, epsilon_y, gamma_xy")
sigma_x, sigma_y, tau_xy = symbols("sigma_x, sigma_y, tau_xy")


def multiple_matmul(*args):
    """
    Perform multiple matrix multiplications in given order.
    :param args: Matrices to multiply
    :return: Result of the consecutive matrix multiplications
    """
    result = args[0]
    for i in range(1, len(args)):
        result = np.matmul(result, args[i])
    return result


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
    New_Composite = "New Composite"

    @property
    def q_3x3_matrix(self):
        Q11 = self.properties['Q11']
        Q22 = self.properties['Q22']
        Q12 = self.properties['Q12']
        Q66 = self.properties['Q66']
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    @property
    def properties(self):
        if self == CompositeType.Glass_Epoxy:
            properties_dico = {'E1': 50 * 10**9, 'E2': 15.2 * 10**9, 'nu_12': 0.254, 'Q66': 4.70 * 10**9,
                               'nu_13': 0.254, 'nu_23': 0.428}
        elif self == CompositeType.Graphite_Epoxy:
            properties_dico = {'E1': 155 * 10**9, 'E2': 12.1 * 10**9, 'nu_12': 0.248, 'Q66': 4.40 * 10**9,
                               'nu_13': 0.248, 'nu_23': 0.458}
        else:
            raise ValueError("Invalid Composite Type")
        nu_21 = (properties_dico['E2'] / properties_dico['E1']) * properties_dico['nu_12']
        Q11 = properties_dico['E1'] / (1 - properties_dico['nu_12'] * nu_21)
        Q22 = properties_dico['E2'] / (1 - properties_dico['nu_12'] * nu_21)
        Q12 = properties_dico['nu_12'] * properties_dico['E2'] / (1 - properties_dico['nu_12'] * nu_21)
        new_values = {'Q11': Q11, 'Q22': Q22, 'Q12': Q12, 'nu_21': nu_21}
        properties_dico.update(new_values)
        return properties_dico

    def expansion_coeffs(self, expansion_type: ExpansionType):
        if self == CompositeType.Graphite_Epoxy:
            dico = {ExpansionType.Thermal: {'alpha_1': -0.018 * 10**-6, 'alpha_2': 24.3 * 10**-6, 'alpha_3': 24.3 * 10**-6},
                    ExpansionType.Hygroscopic: {'beta_1': 146 * 10**-6, 'beta_2': 4770 * 10**-6, 'beta_3': 4770 * 10**-6}}
        elif self == CompositeType.Glass_Epoxy:
            dico = {ExpansionType.Thermal: {'alpha_1': 6.34 * 10**-6, 'alpha_2': 23.3 * 10**-6, 'alpha_3': 23.3 * 10**-6},
                    ExpansionType.Hygroscopic: {'beta_1': 434 * 10**-6, 'beta_2': 6320 * 10**-6, 'beta_3': 6320 * 10**-6}}
        elif self == CompositeType.New_Composite:
            return None
        else:
            raise ValueError("Invalid Composite Type")
        return dico[expansion_type]

    @property
    def s_3x3_matrix(self):
        return np.linalg.inv(self.q_3x3_matrix)


class Composite:
    """
    Composite Class to solve for the undefined variables in a Composite Material.
    Accepts the composite type, the angle of the fibers, the temperature delta and the moisture delta.
    """
    def __init__(self, composite_type: CompositeType, angle: int = 0, delta_t: int | float = 0, delta_m: int | float = 0):
        """
        Initialize the Composite Class.
        Only argument required is the Composite Type.
        Fiber angle, temperature delta and humidity delta are optional arguments.
        :param composite_type:
        :param angle:
        :param delta_t:
        :param delta_m:
        """
        self.angle = np.radians(angle)
        self.composite_type = composite_type
        self.variables = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.variables_to_solve = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.delta_t = delta_t
        self.delta_m = delta_m
        if composite_type != CompositeType.New_Composite:
            self.s_3x3_matrix = composite_type.s_3x3_matrix
            self.q_3x3_matrix = composite_type.q_3x3_matrix
        else:
            self.s_3x3_matrix = None
            self.q_3x3_matrix = None
        self.global_s_matrix_cache = None

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
    def global_s_matrix(self) -> Matrix:
        """
        S Matrix in the global referential.
        """
        if self.global_s_matrix_cache is not None:
            return self.global_s_matrix_cache
        else:
            return multiple_matmul(self.t_transposed_matrix, self.s_3x3_matrix, self.t_matrix)

    @property
    def global_q_matrix(self) -> np.ndarray:
        """
        Q Matrix in the global referential.
        """
        return np.linalg.inv(self.global_s_matrix)

    @property
    def global_thermal_coeffs(self) -> tuple:
        """
        Thermal coefficients in the global referential.
        :return: alpha_x, alpha_y, alpha_xy, alpha_z
        """
        thermal_coeffs = self.composite_type.expansion_coeffs(ExpansionType.Thermal)
        alpha_1, alpha_2, alpha_3 = thermal_coeffs['alpha_1'], thermal_coeffs['alpha_2'], thermal_coeffs['alpha_3']
        alpha_x = alpha_1 * np.cos(self.angle) ** 2 + alpha_2 * np.sin(self.angle) ** 2
        alpha_y = alpha_1 * np.sin(self.angle) ** 2 + alpha_2 * np.cos(self.angle) ** 2
        alpha_xy = 2 * (alpha_1 - alpha_2) * np.sin(self.angle) * np.cos(self.angle)
        alpha_z = alpha_3
        return alpha_x, alpha_y, alpha_xy, alpha_z

    @property
    def global_hygroscopic_coeffs(self) -> tuple:
        """
        Hygroscopic coefficients in the global referential.
        :return: beta_x, beta_y, beta_xy, beta_z
        """
        hygroscopic_coeffs = self.composite_type.expansion_coeffs(ExpansionType.Hygroscopic)
        beta_1, beta_2, beta_3 = hygroscopic_coeffs['beta_1'], hygroscopic_coeffs['beta_2'], hygroscopic_coeffs['beta_3']
        beta_x = beta_1 * np.cos(self.angle) ** 2 + beta_2 * np.sin(self.angle) ** 2
        beta_y = beta_1 * np.sin(self.angle) ** 2 + beta_2 * np.cos(self.angle) ** 2
        beta_xy = 2 * (beta_1 - beta_2) * np.sin(self.angle) * np.cos(self.angle)
        beta_z = beta_3
        return beta_x, beta_y, beta_xy, beta_z

    def update_variables(self, values: tuple, stress: bool = True) -> list:
        """
        Update the variables to solve for the undefined variables.
        If a value is given, remove the corresponding variable from the list of variables to solve.
        If value is None, returns the symbol for sympy to solve.
        :param values: (sigma_x, sigma_y, tau_xy) or (epsilon_x, epsilon_y, gamma_xy)
        :param stress: True if the values are stresses, False if they are strains
        :return: Updated list of values with symbols or values
        """
        updated_values = []
        for index, value in enumerate(values):
            if stress:
                index = index + 3
            if value is not None:
                self.variables_to_solve.remove(self.variables[index])
            else:
                value = self.variables[index]
            updated_values.append(value)
        return updated_values

    def stress_matrix(self, values: tuple = (sigma_x, sigma_y, tau_xy)) -> Matrix:
        variables = self.update_variables(values, stress=True)
        matrix = Matrix([[variables[0]], [variables[1]], [variables[2]]])*10**6
        return matrix

    def strain_matrix(self, values: tuple = (epsilon_x, epsilon_y, gamma_xy)) -> Matrix:
        variables = self.update_variables(values, stress=False)
        total_matrix = None
        for index, variable in enumerate(variables):
            matrix = Matrix(
                [variable*10**-6 - (self.global_thermal_coeffs[index] * self.delta_t) - (self.global_hygroscopic_coeffs[index] * self.delta_m)])
            if index == 0:
                total_matrix = matrix
            else:
                total_matrix = total_matrix.row_insert(index, matrix)
        return total_matrix

    def solve(self, strains: tuple, stresses: tuple) -> dict:
        """
        Solve for the Composite's undefined variables
        :param strains: epsilon_x (mu_epsilon), epsilon_y (mu_epsilon), gamma_xy (mu_rad)
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: Dictionary of the solved variables
        """
        strain_matrix = self.strain_matrix(strains)
        stress_matrix = self.stress_matrix(stresses)
        equation = Eq(Matrix(self.global_s_matrix) * stress_matrix, strain_matrix)
        solution = solve(equation, self.variables_to_solve)
        return solution

    def epsilon_3(self, stresses: tuple):
        """
        Calculate the third strain component from the given stresses.
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: epsilon_3 (mu_epsilon)
        """
        properties = self.composite_type.properties
        S_13 = -properties['nu_13']/properties['E1']
        S_23 = -properties['nu_23']/properties['E2']
        return (S_13*(stresses[0]*10**6) + S_23*(stresses[1]*10**6)) / 10**6

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
        pressure = pressure * 10**6
        sigma_a = pressure * r / thickness*2
        sigma_h = pressure * r / thickness
        return {'sigma_a': sigma_a / 10**6, 'sigma_h': sigma_h / 10**6}

    def __str__(self):
        return f"{self.strain_matrix(values=(10, 0, None))} = {self.global_s_matrix}*{self.stress_matrix(values=(10, 0, None))}"


if __name__ == '__main__':
    # Question 1
    composite = Composite(angle=0, composite_type=CompositeType.Glass_Epoxy, delta_t=75)
    print(CompositeType.Glass_Epoxy.q_3x3_matrix/10**9)
    print(CompositeType.Glass_Epoxy.s_3x3_matrix/10**-12)
    print(composite.solve(strains=(None, None, None), stresses=(20, 10, -5)))
    print(composite.epsilon_3(stresses=(20, 10, -5)))

    # Question 2
    radial_composite = Composite(angle=53, composite_type=CompositeType.Graphite_Epoxy, delta_t=-150, delta_m=1)
    radial_stresses = radial_composite.solve_radial_stresses(pressure=1.2, diameter=0.5, thickness=8)
    print(radial_stresses)
    print(tuple(coeff/10**-6 for coeff in radial_composite.global_hygroscopic_coeffs))
    print(tuple(coeff/10**-6 for coeff in radial_composite.global_thermal_coeffs))
    print(radial_composite.global_s_matrix/10**-12)
    print(radial_composite.global_q_matrix/10**9)
    print(radial_composite.solve(strains=(None, None, None),
                                 stresses=(radial_stresses['sigma_a'], radial_stresses['sigma_h'], 0)))

    #test solver
    test_composite = Composite(angle=30, composite_type=CompositeType.Graphite_Epoxy)
    print(test_composite.solve(strains=(1000, 0, 0), stresses=(None, None, None)))
