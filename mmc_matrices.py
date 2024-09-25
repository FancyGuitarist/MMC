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


class ExpansionCoeffs:
    """
    Expansion Coefficients for Thermal and Hydric Expansion
    """
    Thermal = np.array([-0.018, 24.3, 24.3]) * (10 ** -6)
    Hydric = np.array([146, 4770, 4770]) * (10 ** -6)

    @property
    def value(self):
        return self.value


class CompositeType(StrEnum):
    """
    Enum for Composite Types.
    Can return the reduced Q and S matrices for the given Composite Type.
    """
    Glass_Epoxy = "Glass Epoxy"
    Graphite_Epoxy = "Graphite Epoxy"

    @property
    def q_3x3_matrix(self):
        if self == CompositeType.Glass_Epoxy:
            E1 = 50 * 10**9
            E2 = 15.2 * 10**9
            nu_12 = 0.254
            nu_21 = (E2 / E1) * nu_12
            Q11 = E1 / (1 - nu_12 * nu_21)
            Q22 = E2 / (1 - nu_12 * nu_21)
            Q12 = nu_12 * E2 / (1 - nu_12 * nu_21)
            Q66 = 4.70 * 10**9
        elif self == CompositeType.Graphite_Epoxy:
            E1 = 155 * 10 ** 9
            E2 = 12.1 * 10 ** 9
            nu_12 = 0.248
            nu_21 = (E2 / E1) * nu_12
            Q11 = E1 / (1 - nu_12 * nu_21)
            Q22 = E2 / (1 - nu_12 * nu_21)
            Q12 = nu_12 * E2 / (1 - nu_12 * nu_21)
            Q66 = 4.40 * 10 ** 9
        else:
            raise ValueError("Invalid Composite Type")
        return np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

    @property
    def s_3x3_matrix(self):
        return np.linalg.inv(self.q_3x3_matrix)


class Composite:
    """
    Composite Class to solve for the undefined variables in a Composite Material.
    Accepts the composite type, the angle of the fibers, the temperature delta and the moisture delta.
    """
    def __init__(self, composite_type: CompositeType, angle: int = 0, delta_t: int = 0, delta_m: int = 0):
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
        self.variables = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.variables_to_solve = [epsilon_x, epsilon_y, gamma_xy, sigma_x, sigma_y, tau_xy]
        self.delta_t = delta_t
        self.delta_m = delta_m
        self.s_3x3_matrix = composite_type.s_3x3_matrix
        self.q_3x3_matrix = composite_type.q_3x3_matrix
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
            return Matrix(multiple_matmul(self.t_transposed_matrix, self.s_3x3_matrix, self.t_matrix))

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
        alpha_1, alpha_2, alpha_3 = ExpansionCoeffs.Thermal
        alpha_x = alpha_1 * np.cos(self.angle) ** 2 + alpha_2 * np.sin(self.angle) ** 2
        alpha_y = alpha_1 * np.sin(self.angle) ** 2 + alpha_2 * np.cos(self.angle) ** 2
        alpha_xy = 2 * (alpha_1 - alpha_2) * np.sin(self.angle) * np.cos(self.angle)
        alpha_z = alpha_3
        return alpha_x, alpha_y, alpha_xy, alpha_z

    @property
    def global_hydric_coeffs(self) -> tuple:
        """
        Hydric coefficients in the global referential.
        :return: beta_x, beta_y, beta_xy, beta_z
        """
        beta_1, beta_2, beta_3 = ExpansionCoeffs.Hydric
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
        matrix = Matrix([[variables[0]], [variables[1]], [variables[2]]]) * (10 ** 6)
        return matrix

    def strain_matrix(self, values: tuple = (epsilon_x, epsilon_y, gamma_xy)) -> Matrix:
        variables = self.update_variables(values, stress=False)
        total_matrix = None
        for index, variable in enumerate(variables):
            matrix = Matrix(
                [variable - self.global_thermal_coeffs[index] * self.delta_t - self.global_hydric_coeffs[index] * self.delta_m])
            if index == 0:
                total_matrix = matrix
            else:
                total_matrix = total_matrix.row_insert(index, matrix)
        return total_matrix * (10 ** (-6))

    def solve(self, strains: tuple, stresses: tuple) -> dict:
        """
        Solve for the Composite's undefined variables
        :param strains: epsilon_x, epsilon_y, gamma_xy
        :param stresses: sigma_x, sigma_y, tau_xy
        :return: Dictionary of the solved variables
        """
        strain_matrix = self.strain_matrix(strains)
        stress_matrix = self.stress_matrix(stresses)
        equation = Eq(self.global_s_matrix * stress_matrix, strain_matrix)
        solution = solve(equation, self.variables_to_solve)
        return solution

    def __str__(self):
        return f"{self.strain_matrix(values=(10, 0, None))} = {self.global_s_matrix}*{self.stress_matrix(values=(10, 0, None))}"


if __name__ == '__main__':
    composite = Composite(angle=0, composite_type=CompositeType.Glass_Epoxy, delta_t=75)
    print(CompositeType.Glass_Epoxy.q_3x3_matrix)
    print(CompositeType.Glass_Epoxy.s_3x3_matrix)
    print(composite.solve(strains=(None, None, None), stresses=(20, 10, -5)))
