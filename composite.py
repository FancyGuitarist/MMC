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
            properties_dico = {'E1': 50 * 10 ** 9, 'E2': 15.2 * 10 ** 9, 'nu_12': 0.254, 'Q66': 4.70 * 10 ** 9,
                               'nu_13': 0.254, 'nu_23': 0.428}
        elif self == CompositeType.Graphite_Epoxy:
            properties_dico = {'E1': 155 * 10 ** 9, 'E2': 12.1 * 10 ** 9, 'nu_12': 0.248, 'Q66': 4.40 * 10 ** 9,
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

    @property
    def safety_properties(self) -> dict:
        """
        Return the safety properties of the Composite Type as a dictionary.
        :return:
        """
        if self == CompositeType.Glass_Epoxy:
            safety_properties = {'sigma_1t': 1000 * 10 ** 6, 'sigma_1c': -600 * 10 ** 6, 'sigma_2t': 30 * 10 ** 6,
                                 'sigma_2c': -120 * 10 ** 6, 'tau_12f': 70 * 10 ** 6}
        elif self == CompositeType.Graphite_Epoxy:
            safety_properties = {'sigma_1t': 1500 * 10 ** 6, 'sigma_1c': -1250 * 10 ** 6, 'sigma_2t': 50 * 10 ** 6,
                                 'sigma_2c': -200 * 10 ** 6, 'tau_12f': 100 * 10 ** 6}
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
            dico = {ExpansionType.Thermal: {'alpha_1': -0.018 * 10 ** -6, 'alpha_2': 24.3 * 10 ** -6,
                                            'alpha_3': 24.3 * 10 ** -6},
                    ExpansionType.Hygroscopic: {'beta_1': 146 * 10 ** -6, 'beta_2': 4770 * 10 ** -6,
                                                'beta_3': 4770 * 10 ** -6}}
        elif self == CompositeType.Glass_Epoxy:
            dico = {ExpansionType.Thermal: {'alpha_1': 6.34 * 10 ** -6, 'alpha_2': 23.3 * 10 ** -6,
                                            'alpha_3': 23.3 * 10 ** -6},
                    ExpansionType.Hygroscopic: {'beta_1': 434 * 10 ** -6, 'beta_2': 6320 * 10 ** -6,
                                                'beta_3': 6320 * 10 ** -6}}
        else:
            raise ValueError("Invalid Composite Type")
        return dico[expansion_type]


class Composite:
    """
    Composite Class to solve for the undefined variables in a Composite Material.
    Accepts the composite type, the angle of the fibers, the temperature delta and the moisture delta.
    """

    def __init__(self, composite_type: CompositeType, angle: int = 0, delta_t: int | float = 0,
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
        self.variables = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.variables_to_solve = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.delta_t = delta_t
        self.delta_m = delta_m

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
        return multiple_matmul(self.t_transposed_matrix, self.composite_type.s_3x3_matrix, self.t_matrix)

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
        :return: alpha_x, alpha_y, alpha_xy, alpha_z
        """
        thermal_coeffs = self.composite_type.expansion_coeffs(ExpansionType.Thermal)
        alpha_1, alpha_2, alpha_3 = thermal_coeffs['alpha_1'], thermal_coeffs['alpha_2'], thermal_coeffs['alpha_3']
        return self.local_to_global_coeffs((alpha_1, alpha_2, alpha_3))

    @property
    def global_hygroscopic_coeffs(self) -> np.ndarray:
        """
        Hygroscopic coefficients in the global referential.
        :return: beta_x, beta_y, beta_xy, beta_z
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
        matrix = Matrix([[variables[0]], [variables[1]], [variables[2]]]) * 10 ** 6
        return matrix

    def strain_matrix(self, values: tuple = (epsilon_x, epsilon_y, gamma_xy)) -> Matrix:
        variables = self.update_variables(values, stress=False)
        total_matrix = None
        for index, variable in enumerate(variables):
            matrix = Matrix(
                [variable * 10 ** -6 - (self.global_thermal_coeffs[index] * self.delta_t) - (
                            self.global_hygroscopic_coeffs[index] * self.delta_m)])
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
        S_13 = -properties['nu_13'] / properties['E1']
        S_23 = -properties['nu_23'] / properties['E2']
        return S_13 * (stresses[0] * 10 ** 6) + S_23 * (stresses[1] * 10 ** 6)

    def epsilon_z(self, stresses: tuple):
        """
        Calculate the third strain component from the given stresses.
        :param stresses: sigma_x (MPa), sigma_y (MPa), tau_xy (MPa)
        :return: epsilon_3 (mu_epsilon)
        """
        beta_3 = self.composite_type.expansion_coeffs(ExpansionType.Hygroscopic)['beta_3']
        alpha_3 = self.composite_type.expansion_coeffs(ExpansionType.Thermal)['alpha_3']
        properties = self.composite_type.properties
        S_13 = -properties['nu_13'] / properties['E1']
        S_23 = -properties['nu_23'] / properties['E2']
        epsilon_z = (((alpha_3 * self.delta_t + beta_3 * self.delta_m
                       + (S_13 * np.cos(self.angle) ** 2 + S_23 * np.sin(self.angle) ** 2) * (stresses[0] * 10 ** 6))
                      + (S_13 * np.sin(self.angle) ** 2 + S_23 * np.cos(self.angle) ** 2) * (stresses[1] * 10 ** 6))
                     + 2 * (S_13 - S_23) * np.sin(self.angle) * np.cos(self.angle) * (stresses[2] * 10 ** 6))
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
        pressure = pressure * 10 ** 6
        sigma_a = (pressure * r) / (thickness * 2)
        sigma_h = pressure * r / thickness
        return {'sigma_a': sigma_a / 10 ** 6, 'sigma_h': sigma_h / 10 ** 6}

    def mechanical_strains(self, values: tuple = (epsilon_x, epsilon_y, gamma_xy)):
        mec_strains = []
        for index, strain in enumerate(values):
            mec_strains.append(strain * 10 ** -6 - (self.global_thermal_coeffs[index] * self.delta_t) - (
                        self.global_hygroscopic_coeffs[index] * self.delta_m))
        return np.array(mec_strains)

    def global_to_local_strains(self, values: tuple):
        return multiple_matmul(self.r_matrix, self.t_matrix, np.linalg.inv(self.r_matrix), values)

    def global_to_local_stresses(self, values: tuple):
        return np.matmul(self.t_matrix, values)

    def fs_max(self, values: tuple):
        """
        Calculate security factor for maximum stress.
        :param values: sigma_1, sigma_2, tau_12 (MPa)
        :return: Fs_max
        """
        properties = self.composite_type.safety_properties
        sigma_1, sigma_2, tau_12 = values
        sigma_1, sigma_2, tau_12 = sigma_1 * 10 ** 6, sigma_2 * 10 ** 6, tau_12 * 10 ** 6
        sigma_1t, sigma_1c = properties['sigma_1t'], properties['sigma_1c']
        sigma_2t, sigma_2c, tau_12f = properties['sigma_2t'], properties['sigma_2c'], properties['tau_12f']
        if sigma_1 > 0:
            sigma_1R = sigma_1t
        else:
            sigma_1R = sigma_1c
        if sigma_2 > 0:
            sigma_2R = sigma_2t
        else:
            sigma_2R = sigma_2c
        Fs_max = symbols('Fs_max')
        eq1 = Eq(1, Fs_max ** 2 * (
                    (sigma_1 / sigma_1R) ** 2 + (sigma_2 / sigma_2R) ** 2 - (sigma_1 * sigma_2 / sigma_1R ** 2) + (
                        tau_12 / tau_12f) ** 2))
        solution_max = solve(eq1, Fs_max)
        Fs_Tsai_Hill = symbols('Fs_Tsai_Hill')
        F1 = (1 / sigma_1t + 1 / sigma_1c)
        F11 = -1 / (sigma_1t * sigma_1c)
        F66 = (1 / tau_12f) ** 2
        F2 = (1 / sigma_2t + 1 / sigma_2c)
        F22 = -1 / (sigma_2t * sigma_2c)
        F12 = -0.5 * np.sqrt(F11 * F22)
        a = F11 * sigma_1 ** 2 + F22 * sigma_2 ** 2 + F66 * tau_12 ** 2 - sigma_1 * sigma_2 * np.sqrt(F11 * F22)
        b = F1 * sigma_1 + F2 * sigma_2
        eq2 = Eq(1, a * Fs_Tsai_Hill ** 2 + b * Fs_Tsai_Hill)
        solution_tsai_hill = solve(eq2, Fs_Tsai_Hill)
        return solution_max, solution_tsai_hill

    def __str__(self):
        return f"{self.strain_matrix(values=(10, 0, None))} = {self.global_s_matrix}*{self.stress_matrix(values=(10, 0, None))}"


if __name__ == '__main__':
    # Question 1
    composite = Composite(angle=0, composite_type=CompositeType.Glass_Epoxy, delta_t=75)
    print(CompositeType.Glass_Epoxy.q_3x3_matrix / 10 ** 9)
    print(CompositeType.Glass_Epoxy.s_3x3_matrix / 10 ** -12)
    print("déformations=", composite.solve(strains=(None, None, None), stresses=(20, 10, -5)))
    print("epsilon_3=", composite.epsilon_3(stresses=(20, 10, -5)))

    # Question 2
    radial_composite = Composite(angle=53, composite_type=CompositeType.Graphite_Epoxy, delta_t=-150, delta_m=1)
    radial_stresses = radial_composite.solve_radial_stresses(pressure=1.2, diameter=0.5, thickness=8)
    print(radial_stresses)
    print("Coefficients:")
    print(np.array(radial_composite.global_hygroscopic_coeffs) / 10 ** -6)
    print(np.array(radial_composite.global_thermal_coeffs) / 10 ** -6)
    print("Matrices S et Q:")
    print(radial_composite.global_s_matrix / 10 ** -12)
    print(radial_composite.global_q_matrix / 10 ** 9)
    print(radial_composite.solve(strains=(None, None, None),
                                 stresses=(radial_stresses['sigma_a'], radial_stresses['sigma_h'], 0)))
    print(
        radial_composite.mechanical_strains(values=(1158.93789983545, 2333.70349172889, -3333.67605465999)) / 10 ** -6)
    print(radial_composite.global_to_local_strains((1158.93789983545, 489.807127432113, -4387.74781007873)))
    print(radial_composite.global_to_local_strains((387.53552459, 1831.40586697, -2395.19626092)))
    print(radial_composite.epsilon_z(stresses=(18.75, 37.5, 0)) / 10 ** -6)
    sigma_1, sigma_2, tau_12 = radial_composite.global_to_local_stresses((18.75, 37.5, 0))
    print(sigma_1, sigma_2, tau_12)
    fs_max, fs_tsai = radial_composite.fs_max((sigma_1, sigma_2, tau_12))
    print(fs_max, fs_tsai)
