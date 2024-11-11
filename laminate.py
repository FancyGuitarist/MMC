import sympy
import re
from composite import Composite, CompositeType, Variables
from sympy import Eq, solve
from sympy.matrices import Matrix
import numpy as np
import pandas as pd


class Laminate:
    def __init__(self, thetas: list, composite_type: CompositeType, h: float = 0.150):
        """
        Laminate class that builds a laminate from a list of angles and a composite type
        :param thetas: list of angles for the laminate layers in degrees
        :param composite_type: CompositeType
        :param h: Thickness of the laminate in mm
        """
        self.thetas = thetas
        self.composite_type = composite_type
        self.h = h / 1000
        self.composites = [Composite(angle=theta, composite_type=composite_type) for theta in thetas]
        self.abd_matrix_cache, self.inv_abd_cache = None, None
        self.global_stress_layers_cache = [None, None]

    @property
    def q_k(self):
        q_k = None
        for composite in self.composites:
            if q_k is None:
                q_k = composite.global_q_matrix
            else:
                q_k += composite.global_q_matrix
        return q_k

    def setup_n_H_z(self):
        n = len(self.thetas)
        H = n * self.h
        z = np.zeros((n + 1))
        z[0] = -H / 2
        for i in range(1, len(z)):
            z[i] = z[i - 1] + self.h
        return n, H, z

    @property
    def abd_matrix(self):
        if self.abd_matrix_cache is not None:
            return self.abd_matrix_cache
        n, H, z = self.setup_n_H_z()

        a_matrix = np.zeros((3, 3))
        b_matrix = np.zeros((3, 3))
        d_matrix = np.zeros((3, 3))

        for i in range(n):
            q_matrix = self.composites[i].global_q_matrix
            a_matrix = a_matrix + q_matrix * (z[i + 1] - z[i])
            b_matrix = b_matrix + q_matrix * (z[i + 1] ** 2 - z[i] ** 2) / 2
            d_matrix = d_matrix + q_matrix * (z[i + 1] ** 3 - z[i] ** 3) / 3
        self.abd_matrix_cache = a_matrix, b_matrix, d_matrix
        return np.block([[a_matrix, b_matrix], [b_matrix, d_matrix]])

    @property
    def inv_abd_matrix(self):
        if self.inv_abd_cache is not None:
            return self.inv_abd_cache
        self.inv_abd_cache = np.linalg.inv(self.abd_matrix)
        return self.inv_abd_cache

    @property
    def a_matrix(self):
        return self.abd_matrix[:3, :3]

    @property
    def inv_a_matrix(self):
        return self.inv_abd_matrix[:3, :3]

    @property
    def b_matrix(self):
        return self.abd_matrix[:3, 3:]

    @property
    def inv_b_matrix(self):
        return self.inv_abd_matrix[:3, 3:]

    @property
    def d_matrix(self):
        return self.abd_matrix[3:, 3:]

    @property
    def inv_d_matrix(self):
        return self.inv_abd_matrix[3:, 3:]

    @property
    def effective_properties(self):
        n = len(self.thetas)
        H = n * self.h
        a_11 = self.inv_a_matrix[0, 0]
        a_12 = self.inv_a_matrix[1, 0]
        a_22 = self.inv_a_matrix[1, 1]
        a_66 = self.inv_a_matrix[2, 2]
        eff_E_x = 1 / (H * a_11)
        eff_E_y = 1 / (H * a_22)
        eff_G_xy = 1 / (H * a_66)
        eff_nu_xy = -a_12 / a_11
        eff_nu_yx = -a_12 / a_22
        eff_properties = {"E_x": eff_E_x, "E_y": eff_E_y, "G_xy": eff_G_xy, "nu_xy": eff_nu_xy, "nu_yx": eff_nu_yx}
        return eff_properties

    def get_variables_to_solve(self, values: list):
        variables = []
        for value in values:
            if isinstance(value, sympy.Symbol):
                variables.append(value)
        return variables

    def adjust_solution_units(self, solution: dict):
        results = {}
        for key, val in solution.items():
            if key in Variables.default_eps:
                val *= 1e6
            if key in Variables.default_N:
                val /= 1e3
            results[key] = round(val, 3)
        return results

    def solve_eps_kap_n_m(self,
                          epsilons: list[Variables | float] = Variables.default_eps,
                          kappas: list[Variables | float] = Variables.default_kap,
                          ns: list[Variables | float] = Variables.default_N,
                          ms: list[Variables | float] = Variables.default_M):
        variables_to_solve = self.get_variables_to_solve(epsilons + kappas + ns + ms)
        eps_kap, n_m = Matrix(epsilons + kappas), Matrix(ns + ms)
        equation = Eq(Matrix(self.inv_abd_matrix) * n_m, eps_kap)
        solution = solve(equation, variables_to_solve)
        return self.adjust_solution_units(solution)

    def global_stress_layers(self, epsilons: list, kappas: list):
        if self.global_stress_layers_cache[0] is not None and self.global_stress_layers_cache[0] == [epsilons, kappas]:
            return self.global_stress_layers_cache[0]
        n, H, z = self.setup_n_H_z()
        eps = Matrix(epsilons) / 1e6
        kap = Matrix(kappas)
        global_stress_layers = {}

        for i in range(n):
            q_matrix = self.composites[i].global_q_matrix
            eps_kap = eps + kap * z[i]
            stress_mat = Matrix(Variables.default_stresses)
            equation = Eq(Matrix(q_matrix) * eps_kap, stress_mat)
            solved = solve(equation, Variables.default_stresses)
            global_stress_layers[f"z_{i}"] = {key: round(solved[key]/1e6, 4) for key in solved}
        self.global_stress_layers_cache = [global_stress_layers, [epsilons, kappas]]
        return self.global_stress_layers_cache[0]

    def local_stress_layers(self, epsilons: list, kappas: list):
        global_layers = self.global_stress_layers(epsilons, kappas)
        sigma_1, sigma_2, tau_12 = Variables.default_local_stresses
        local_layers = {}
        for key in global_layers.keys():
            match = re.search(r"z_(\d+)", key)
            index = int(match.group(1))
            t_matrix = self.composites[index].t_matrix
            global_layer = np.vstack([value for value in global_layers[key].values()])
            local_layer = t_matrix @ global_layer
            local_layers[key] = {sigma_1: round(float(local_layer[0]), 4),
                                 sigma_2: round(float(local_layer[1]), 4),
                                 tau_12: round(float(local_layer[2]), 4)}
        return local_layers

    def display_effective_properties(self):
        eff_properties = self.effective_properties
        for key, value in eff_properties.items():
            if key == "G_xy" or key == "E_x" or key == "E_y":
                print(f"{key}: {value / 1e9:.2f} GPa")
            else:
                print(f"{key}: {value:.4f}")
