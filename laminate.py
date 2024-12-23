import sympy
import re
from composite import *
from sympy import Eq, solve
from sympy.matrices import Matrix
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import StrEnum

p_symbol = symbols("p")
fs_symbol = symbols("f_s")


class FailureCriteria(StrEnum):
    TsaiHill = "Tsai-Hill"
    TsaiWu = "Tsai-Wu"
    MaxCriteria = "Maximum Stress Criteria"


class RupturePressure:
    def __init__(self, eps_kap: dict,  p: float | sympy.Symbol, composite: Composite, deltas: tuple[float, float] = (0, 0)):
        self.p = p
        self.composite = composite
        self.eps_kap = eps_kap
        self.delta_t, self.delta_m = deltas
        self.f_s = 1
        self.compression = False

    def get_sigmas_eq(self):
        comp_thermal_coeffs = Matrix(self.composite.global_thermal_coeffs[:3])
        comp_hygroscopic_coeffs = Matrix(self.composite.global_hygroscopic_coeffs[:3])
        sigmas = (Matrix(self.composite.global_q_matrix) *
                  (Matrix(self.eps_kap[:3]) - (comp_thermal_coeffs * self.delta_t) - (comp_hygroscopic_coeffs * self.delta_m)))
        local_sigmas = Matrix(self.composite.t_matrix) * sigmas
        sigma_1, sigma_2, tau_12 = local_sigmas[0], local_sigmas[1], local_sigmas[2]
        return sigma_1, sigma_2, tau_12

    def tsai_hill_eq(self):
        sigma_1, sigma_2, tau_12 = self.get_sigmas_eq()
        properties = self.composite.composite_type.safety_properties
        sigma_1t, sigma_2t, tau_12f = properties['sigma_1t'], properties['sigma_2t'], properties['tau_12f']
        equation = Eq(1/(self.f_s ** 2), ((sigma_1 / sigma_1t) ** 2 + (sigma_2 / sigma_2t) ** 2 -
                          (sigma_1 * sigma_2 / sigma_1t ** 2) + (tau_12 / tau_12f) ** 2))
        return equation

    def tsai_wu_eq(self):
        sigma_1, sigma_2, tau_12 = self.get_sigmas_eq()
        f_ijs = self.composite.f_ij_elements()
        a = f_ijs["F11"] * sigma_1 ** 2 + f_ijs["F22"] * sigma_2 ** 2 + f_ijs["F66"] * tau_12 ** 2 - np.sqrt(f_ijs["F11"] * f_ijs["F22"]) * sigma_1 * sigma_2
        b = f_ijs["F1"] * sigma_1 + f_ijs["F2"] * sigma_2
        Wu = a * self.f_s ** 2 + b * self.f_s
        equation = Eq(1, Wu)
        return equation

    def max_stress_eq(self):
        sigma_1, sigma_2, tau_12 = self.get_sigmas_eq()
        properties = self.composite.composite_type.safety_properties
        if self.compression:
            sigma_1R, sigma_2R, tau_12f = properties['sigma_1c'], properties['sigma_2c'], -properties['tau_12f']
        else:
            sigma_1R, sigma_2R, tau_12f = properties['sigma_1t'], properties['sigma_2t'], properties['tau_12f']
        eq1 = Eq(sigma_1R / sigma_1, self.f_s)
        eq2 = Eq(sigma_2R / sigma_2, self.f_s)
        eq3 = Eq(tau_12f / tau_12, self.f_s)
        return eq1, eq2, eq3

    def get_equation(self, criteria: FailureCriteria):
        if criteria == FailureCriteria.TsaiHill:
            return self.tsai_hill_eq()
        elif criteria == FailureCriteria.TsaiWu:
            return self.tsai_wu_eq()
        elif criteria == FailureCriteria.MaxCriteria:
            return self.max_stress_eq()

    def solve_pressure_or_f_s(self, criteria: FailureCriteria):
        equation = self.get_equation(criteria)
        if isinstance(self.p, sympy.Symbol):
            variable = self.p
            factor = 1e6
            solve_for = "p"
        elif isinstance(self.f_s, sympy.Symbol):
            variable = self.f_s
            factor = 1
            solve_for = "f_s"
        else:
            raise ValueError("Cannot solve for both p and f_s.")
        if criteria == FailureCriteria.MaxCriteria:
            p1 = solve(equation[0], variable)
            p2 = solve(equation[1], variable)
            p3 = solve(equation[2], variable)
            if solve_for == "p":
                solved = max(p1 + p2 + p3)
            else:
                solved = min(np.abs(p1 + p2 + p3))
        else:
            if self.compression and solve_for == "p":
                solved = min(Matrix(solve(equation, variable)))
            elif not self.compression and solve_for == "p":
                solved = max(Matrix(solve(equation, variable)))
            else:
                solved = max(Matrix(solve(equation, variable)))
        return round(solved / factor, 5)


class LaminateAngles:
    def __init__(self, angles_str: str):
        """
        LaminateAngles class that parses a string of angles into a list of angles.

        Syntax:
        ± : plus or minus, ¬ : non-symmetrical, nS : n repeats with symmetry, _k : repeat character k times
        example:
        "[0_2/±45/¬90]S" -> [0, 0, 45, -45, 90, -45, 45, 0, 0]
        :param angles_str: string of angles separated by commas
        :returns: list of angles
        """
        self.angles_str = angles_str

    def get_angles_list(self, verbose=False) -> list:
        # Step 1: Extract the main pattern and check for symmetry
        main_pattern = r"\[(.+)\](\d*)(S?)"
        main_match = re.match(main_pattern, self.angles_str)
        if not main_match:
            raise ValueError("Invalid angles string format")

        numbers_part, n_global_repeats, symmetry = main_match.groups()
        n_global_repeats = int(n_global_repeats) if n_global_repeats else 1

        # Step 2: Split the content inside brackets by commas and process each part
        segments = numbers_part.split('/')
        result_sequence = []
        single_symmetry, non_sym_len = False, None

        for segment in segments:
            # Step 3: Handle each segment like ±45, 90_2, or 0
            pattern = r"([¬]?\s*[±]?[-]?\d+)(?:_(\d+))?"
            match = re.match(pattern, segment.strip())
            if not match:
                continue

            value_str, repeat_count = match.groups()
            if verbose:
                print(f"Value: {value_str}, Repeat: {repeat_count}")
            value = int(value_str.lstrip('¬').lstrip('±').lstrip('-'))

            # Determine if the value has a ± sign
            if '±' in value_str:
                sequence = [value, -value]
            elif '-' in value_str:
                sequence = [-value]
            else:
                sequence = [value]

            if '¬' in value_str:
                single_symmetry = True
                non_sym_len = len(sequence)

            # Repeat the sequence based on the repeat count (default to 1 if not provided)
            repeat_count = int(repeat_count) if repeat_count else 1
            expanded_sequence = sequence * repeat_count

            # Append the expanded sequence to the result
            result_sequence.extend(expanded_sequence)

        # Step 4: If 'S' is present, make the sequence symmetrical
        if symmetry:
            if single_symmetry and n_global_repeats == 1:
                cut_sequence = result_sequence[:-non_sym_len]
                non_sym_sequence = result_sequence[len(cut_sequence):]
                symmetrical_sequence = cut_sequence + non_sym_sequence + cut_sequence[::-1]
            elif not single_symmetry and n_global_repeats >= 1:
                repeated_sequence = result_sequence * int(n_global_repeats)
                symmetrical_sequence = repeated_sequence + repeated_sequence[::-1]
            else:
                raise ValueError("Invalid format, cannot have both non-symmetrical (¬) and multiple symmetries (kS)")
            final_sequence = symmetrical_sequence
        else:
            final_sequence = result_sequence

        return final_sequence


class Laminate:
    def __init__(self, thetas: LaminateAngles | list, composite_type: CompositeType, h: float = 0.150, delta_t: float = 0,
                 delta_m: float = 0):
        """
        Laminate class that builds a laminate from a list of angles and a composite type
        :param thetas: list of angles for the laminate layers in degrees
        :param composite_type: CompositeType
        :param h: Thickness of the laminate in mm
        :param delta_t: Temperature change in Celsius
        :param delta_m: Moisture change in percentage
        """
        self.thetas = thetas.get_angles_list() if isinstance(thetas, LaminateAngles) else thetas
        self.composite_type = composite_type
        self.h = h / 1000
        self.delta_t, self.delta_m = delta_t, delta_m
        self.composites = [Composite(angle=theta, composite_type=composite_type) for theta in self.thetas]
        self.n_H_z_cache, self.abd_matrix_cache, self.inv_abd_cache = None, None, None
        self.global_stress_layers_cache = [None, None]
        self.local_stress_layers_cache = [None, None]
        self.current_loads = Matrix([0, 0, 0, 0, 0, 0])

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
        if self.n_H_z_cache is not None:
            return self.n_H_z_cache
        n = len(self.thetas)
        H = n * self.h
        z = np.zeros((n + 1))
        z[0] = -H / 2
        for i in range(1, len(z)):
            z[i] = z[i - 1] + self.h
        self.n_H_z_cache = n, H, z
        return self.n_H_z_cache

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
        self.abd_matrix_cache = np.block([[a_matrix, b_matrix], [b_matrix, d_matrix]])
        return self.abd_matrix_cache

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

    def kth_expansion_coefficients(self, q_matrix: np.ndarray, expansion_coefffs: np.ndarray, zs: tuple):
        z1, z0 = zs
        z1_z0 = z1 - z0
        z1_z0_sq = z1 ** 2 - z0 ** 2
        coeff_xk, coeff_yk, coeff_xyk = expansion_coefffs[0], expansion_coefffs[1], expansion_coefffs[2]
        thermal_1 = (q_matrix[0, 0] * coeff_xk) + (q_matrix[0, 1] * coeff_yk) + (q_matrix[0, 2] * coeff_xyk)
        thermal_2 = (q_matrix[1, 0] * coeff_xk) + (q_matrix[1, 1] * coeff_yk) + (q_matrix[1, 2] * coeff_xyk)
        thermal_3 = (q_matrix[2, 0] * coeff_xk) + (q_matrix[2, 1] * coeff_yk) + (q_matrix[2, 2] * coeff_xyk)
        N_xk = thermal_1 * z1_z0
        N_yk = thermal_2 * z1_z0
        N_xyk = thermal_3 * z1_z0
        M_xk = 0.5 * thermal_1 * z1_z0_sq
        M_yk = 0.5 * thermal_2 * z1_z0_sq
        M_xyk = 0.5 * thermal_3 * z1_z0_sq
        return np.array([N_xk, N_yk, N_xyk, M_xk, M_yk, M_xyk])

    @property
    def laminate_expansion_coefficients(self):
        n, H, z = self.setup_n_H_z()
        thermal_matrix = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        hygroscopic_matrix = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        for k in range(n):
            if self.delta_m == 0 and self.delta_t == 0:
                break
            composite_k = self.composites[k]
            q_matrix = composite_k.global_q_matrix
            zs = z[k + 1], z[k]
            if self.delta_t != 0:
                thermal_coeffs = composite_k.global_thermal_coeffs
                thermal_matrix += self.kth_expansion_coefficients(q_matrix, thermal_coeffs, zs)
            if self.delta_m != 0:
                hygroscopic_coeffs = composite_k.global_hygroscopic_coeffs
                hygroscopic_matrix += self.kth_expansion_coefficients(q_matrix, hygroscopic_coeffs, zs)
        return thermal_matrix, hygroscopic_matrix

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

    def add_forces_and_moments(self, ns: list[float], ms: list[float]):
        new_nm = Matrix(ns + ms)
        if self.current_loads is None:
            self.current_loads = new_nm
        else:
            self.current_loads += new_nm
        return self.current_loads

    def solve_eps_kap_n_m(self,
                          epsilons: list[Variables | float] = Variables.default_eps,
                          kappas: list[Variables | float] = Variables.default_kap,
                          ns: list[Variables | float] = Variables.default_N,
                          ms: list[Variables | float] = Variables.default_M):
        variables_to_solve = self.get_variables_to_solve(epsilons + kappas + ns + ms)
        thermal_matrix, hygroscopic_matrix = self.laminate_expansion_coefficients
        eps_kap, n_m = Matrix(epsilons + kappas), Matrix(ns + ms)
        n_m = n_m + self.current_loads + Matrix(thermal_matrix) * self.delta_t + Matrix(hygroscopic_matrix) * self.delta_m
        equation = Eq(Matrix(self.inv_abd_matrix) * n_m, eps_kap)
        solution = solve(equation, variables_to_solve)
        return self.adjust_solution_units(solution)

    def failure_pressure_criteria(self, d: float, criteria: FailureCriteria, fs=None, p=None,
                                  compression: bool = False, use_interior_r: bool = False, verbose: bool = False):
        if p is None and fs is None:
            raise ValueError("Either p or f_s must be provided.")
        if p is None:
            p = p_symbol
        if fs is None:
            fs = fs_symbol
        laminate_thermal_coeffs = self.laminate_expansion_coefficients[0]
        laminate_hygroscopic_coeffs = self.laminate_expansion_coefficients[1]
        results_dict = defaultdict(dict)
        _, H, _ = self.setup_n_H_z()
        if use_interior_r:
            r = (d / 2) - H
        else:
            r = d / 2
        n_ms = Matrix([p * r / 2, p * r, 0, 0, 0, 0]) + self.current_loads
        if verbose:
            print(f"Thermal loads: {np.round((np.array(laminate_thermal_coeffs) * self.delta_t)/1e3, 3)} kN/m")
        eps_kap = (Matrix(self.inv_abd_matrix) *
                   (n_ms + Matrix(laminate_thermal_coeffs) * self.delta_t +
                    Matrix(laminate_hygroscopic_coeffs) * self.delta_m))
        for comp in self.composites:
            angle = int(round(np.degrees(comp.angle)))
            if angle in results_dict:
                continue
            p_current, fs_current = p, fs
            rupture = RupturePressure(eps_kap, p_current, comp, (self.delta_t, self.delta_m))
            rupture.f_s = fs_current
            rupture.compression = compression
            solved = rupture.solve_pressure_or_f_s(criteria)
            if isinstance(p, sympy.Symbol):
                solved_p = solved
                solved_fs = fs_current
            else:
                solved_p = p_current
                solved_fs = solved
            results_dict[angle]["p"] = solved_p
            results_dict[angle]["f_s"] = solved_fs
        return dict(results_dict)

    def solve_residual_stresses(self, eps_kap: dict):
        eps_mat = Matrix([eps_kap[key] * 1e-6 for key in Variables.default_eps])
        results = defaultdict(dict)
        for composite in self.composites:
            alpha_mat = Matrix(composite.global_thermal_coeffs[:3])
            total_eps = eps_mat - (alpha_mat * self.delta_t)
            equation = Eq(Matrix(composite.global_q_matrix) * total_eps, Matrix(Variables.default_stresses))
            solution = solve(equation, Variables.default_stresses)
            results[int(round(np.degrees(composite.angle)))] = {key: round(value / 1e6, 3) for key, value in solution.items()}
        return dict(results)

    def local_residual_stresses(self, eps_kap: dict):
        global_results = self.solve_residual_stresses(eps_kap)
        local_results = defaultdict(dict)
        for index, theta in enumerate(self.thetas):
            composite = self.composites[index]
            t_matrix = composite.t_matrix
            global_stresses = np.vstack([value for value in global_results[theta].values()])
            local_stresses = t_matrix @ global_stresses
            local_results[theta] = {key: round(float(value), 3) for key, value in zip(Variables.default_local_stresses, local_stresses)}
        return dict(local_results)

    def curvature(self, x: np.ndarray, y: np.ndarray, eps_kap: dict):
        kap_x, kap_y, kap_xy = eps_kap[Variables.kap_x], eps_kap[Variables.kap_y], eps_kap[Variables.kap_xy]
        return -0.5 * ((kap_x * y ** 2) + (kap_y * x ** 2) + (kap_xy * x * y))

    def plot_curvature(self, eps_kap: dict, plate_dimensions: tuple = (0.2, 0.2)):
        length, width = plate_dimensions
        x = np.linspace(-length / 2, length / 2, 100)
        y = np.linspace(-width / 2, width / 2, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.curvature(X, Y, eps_kap)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.invert_yaxis()
        ax.set_xlabel('X') # Swapping X and Y axis to get a better view and match GMC-4250 course's convention
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        print(f"Max curvature: {np.max(Z):.4f}")

    def global_stress_layers(self, epsilons: list, kappas: list):
        if self.global_stress_layers_cache[0] is not None and self.global_stress_layers_cache[0] == [epsilons, kappas]:
            return self.global_stress_layers_cache[0]
        n, H, z = self.setup_n_H_z()
        eps = Matrix(epsilons) / 1e6
        kap = Matrix(kappas)
        stress_mat = Matrix(Variables.default_stresses)
        global_stress_layers = defaultdict(dict)

        for i in range(n):
            comp = self.composites[i]
            q_matrix = comp.global_q_matrix
            eps_kap_top = eps + kap * z[i]
            eps_kap_top = eps_kap_top - Matrix(comp.global_thermal_coeffs[:3]) * self.delta_t
            eps_kap_bot = eps + kap * z[i + 1]
            eps_kap_bot = eps_kap_bot - Matrix(comp.global_thermal_coeffs[:3]) * self.delta_t
            equation_top = Eq(Matrix(q_matrix) * eps_kap_top, stress_mat)
            equation_bot = Eq(Matrix(q_matrix) * eps_kap_bot, stress_mat)
            solved_top = solve(equation_top, Variables.default_stresses)
            solved_bot = solve(equation_bot, Variables.default_stresses)
            global_stress_layers[f"z_{i}"]["Top"] = {key: round(solved_top[key] / 1e6, 2) for key in solved_top}
            global_stress_layers[f"z_{i}"]["Bottom"] = {key: round(solved_bot[key] / 1e6, 2) for key in solved_bot}
        self.global_stress_layers_cache = [dict(global_stress_layers), [epsilons, kappas]]
        return self.global_stress_layers_cache[0]

    def local_stress_layers(self, epsilons: list, kappas: list):
        if self.local_stress_layers_cache[0] is not None and self.local_stress_layers_cache[0] == [epsilons, kappas]:
            return self.local_stress_layers_cache[0]
        global_layers = self.global_stress_layers(epsilons, kappas)
        sigma_1, sigma_2, tau_12 = Variables.default_local_stresses
        local_layers = defaultdict(dict)
        for key in global_layers.keys():
            match = re.search(r"z_(\d+)", key)
            index = int(match.group(1))
            t_matrix = self.composites[index].t_matrix
            global_layer_top = np.vstack([value for value in global_layers[key]["Top"].values()])
            global_layer_bot = np.vstack([value for value in global_layers[key]["Bottom"].values()])
            local_layer_top = t_matrix @ global_layer_top
            local_layers[key]["Top"] = {sigma_1: round(float(local_layer_top[0]), 4),
                                        sigma_2: round(float(local_layer_top[1]), 4),
                                        tau_12: round(float(local_layer_top[2]), 4)}
            local_layer_bot = t_matrix @ global_layer_bot
            local_layers[key]["Bottom"] = {sigma_1: round(float(local_layer_bot[0]), 4),
                                           sigma_2: round(float(local_layer_bot[1]), 4),
                                           tau_12: round(float(local_layer_bot[2]), 4)}
        self.local_stress_layers_cache = [dict(local_layers), [epsilons, kappas]]
        return self.local_stress_layers_cache[0]

    def global_strain_layers(self, epsilons: list, kappas: list):
        n, H, z = self.setup_n_H_z()
        eps = Matrix(epsilons) / 1e6
        kap = Matrix(kappas)
        epsilon_x, epsilon_y, gamma_xy = Variables.default_strains
        global_strain_layers = defaultdict(dict)

        for i in range(n):
            eps_kap_top = (eps + kap * z[i]) * 1e6
            eps_kap_bot = (eps + kap * z[i + 1]) * 1e6
            global_strain_layers[f"z_{i}"]["Top"] = {epsilon_x: eps_kap_top[0],
                                                     epsilon_y: eps_kap_top[1],
                                                     gamma_xy: eps_kap_top[2]}
            global_strain_layers[f"z_{i}"]["Bottom"] = {epsilon_x: eps_kap_bot[0],
                                                        epsilon_y: eps_kap_bot[1],
                                                        gamma_xy: eps_kap_bot[2]}
        return dict(global_strain_layers)

    def local_strain_layers(self, epsilons: list, kappas: list):
        global_layers = self.global_strain_layers(epsilons, kappas)
        epsilon_1, epsilon_2, gamma_12 = Variables.default_local_strains
        local_layers = defaultdict(dict)
        for key in global_layers.keys():
            match = re.search(r"z_(\d+)", key)
            index = int(match.group(1))
            t_matrix = self.composites[index].t_matrix
            r_matrix = self.composites[index].r_matrix
            global_layer_top = np.vstack([value for value in global_layers[key]["Top"].values()])
            global_layer_bot = np.vstack([value for value in global_layers[key]["Bottom"].values()])
            local_layer_top = r_matrix @ t_matrix @ np.linalg.inv(r_matrix) @ global_layer_top
            local_layers[key]["Top"] = {epsilon_1: round(float(local_layer_top[0]), 4),
                                        epsilon_2: round(float(local_layer_top[1]), 4),
                                        gamma_12: round(float(local_layer_top[2]), 4)}
            local_layer_bot = r_matrix @ t_matrix @ np.linalg.inv(r_matrix) @ global_layer_bot
            local_layers[key]["Bottom"] = {epsilon_1: round(float(local_layer_bot[0]), 4),
                                           epsilon_2: round(float(local_layer_bot[1]), 4),
                                           gamma_12: round(float(local_layer_bot[2]), 4)}
        return dict(local_layers)

    def global_stress_layer_table(self, epsilons: list, kappas: list, display: bool = True, save_name: str = None):
        sigma_x, sigma_y, tau_xy = Variables.default_stresses
        global_stresses = self.global_stress_layers(epsilons, kappas)
        rows = []
        for i in range(len(self.composites)):
            key = f'z_{i}'
            row = {
                'z': f'{i}',
                'Top sigma_x': global_stresses[key]["Top"][sigma_x],
                'Top sigma_y': global_stresses[key]["Top"][sigma_y],
                'Top tau_xy': global_stresses[key]["Top"][tau_xy],

                'Bottom sigma_x': global_stresses[key]["Bottom"][sigma_x],
                'Bottom sigma_y': global_stresses[key]["Bottom"][sigma_y],
                'Bottom tau_xy': global_stresses[key]["Bottom"][tau_xy],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if display:
            print(df)
        if save_name is not None:
            df.to_csv(f"{save_name}.csv")
        return df

    def global_strain_layer_table(self, epsilons: list, kappas: list, display: bool = True, save_name: str = None):
        epsilon_x, epsilon_y, gamma_xy = Variables.default_strains
        global_strains = self.global_strain_layers(epsilons, kappas)
        rows = []
        for i in range(len(self.composites)):
            key = f'z_{i}'
            row = {
                'z': f'{i}',
                'Top epsilon_x': global_strains[key]["Top"][epsilon_x],
                'Top epsilon_y': global_strains[key]["Top"][epsilon_y],
                'Top gamma_xy': global_strains[key]["Top"][gamma_xy],

                'Bottom epsilon_x': global_strains[key]["Bottom"][epsilon_x],
                'Bottom epsilon_y': global_strains[key]["Bottom"][epsilon_y],
                'Bottom gamma_xy': global_strains[key]["Bottom"][gamma_xy],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if display:
            print(df)
        if save_name is not None:
            df.to_csv(f"{save_name}.csv")
        return df

    def local_stress_layer_table(self, epsilons: list, kappas: list, display: bool = True, save_name: str = None):
        sigma_1, sigma_2, tau_12 = Variables.default_local_stresses
        local_stresses = self.local_stress_layers(epsilons, kappas)
        rows = []
        for i in range(len(self.composites)):
            key = f'z_{i}'
            row = {
                'z': f'{i}',
                'Top sigma_1': local_stresses[key]["Top"][sigma_1],
                'Top sigma_2': local_stresses[key]["Top"][sigma_2],
                'Top tau_12': local_stresses[key]["Top"][tau_12],

                'Bottom sigma_1': local_stresses[key]["Bottom"][sigma_1],
                'Bottom sigma_2': local_stresses[key]["Bottom"][sigma_2],
                'Bottom tau_12': local_stresses[key]["Bottom"][tau_12],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if display:
            print(df)
        if save_name is not None:
            df.to_csv(f"{save_name}.csv")
        return df

    def local_strain_layer_table(self, epsilons: list, kappas: list, display: bool = True, save_name: str = None):
        epsilon_1, epsilon_2, gamma_12 = Variables.default_local_strains
        local_strains = self.local_strain_layers(epsilons, kappas)
        rows = []
        for i in range(len(self.composites)):
            key = f'z_{i}'
            row = {
                'z': f'{i}',
                'Top epsilon_1': local_strains[key]["Top"][epsilon_1],
                'Top epsilon_2': local_strains[key]["Top"][epsilon_2],
                'Top gamma_12': local_strains[key]["Top"][gamma_12],

                'Bottom epsilon_1': local_strains[key]["Bottom"][epsilon_1],
                'Bottom epsilon_2': local_strains[key]["Bottom"][epsilon_2],
                'Bottom gamma_12': local_strains[key]["Bottom"][gamma_12],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if display:
            print(df)
        if save_name is not None:
            df.to_csv(f"{save_name}.csv")
        return df

    def display_effective_properties(self):
        eff_properties = self.effective_properties
        for key, value in eff_properties.items():
            if key == "G_xy" or key == "E_x" or key == "E_y":
                print(f"{key}: {value / 1e9:.2f} GPa")
            else:
                print(f"{key}: {value:.4f}")

    def triple_gauge_global_strains(self, epsilons_abc: dict, thetas: list, width=30, length=342):
        """
        Compute the global strains for a triple gauge array
        :param epsilons_abc: dict of the form {'epsilon_a': [values], 'epsilon_b': [values], 'epsilon_c': [values]}
        :param thetas: list of the gauge angles in degrees
        :param width: width of the laminate in mm
        :param length: length of the laminate in mm
        :return: dictionary of the form {index: {epsilon_x: value, epsilon_y: value, gamma_xy: value}} where the values are in micrometers/meter
        """
        eps_x, eps_y, gam_xy = symbols('epsilon_x epsilon_y gamma_xy')
        w, L = width * 1e-3, length * 1e-3
        theta_a, theta_b, theta_c = thetas
        epsilons_xy_matrix = Matrix([eps_x, eps_y, gam_xy])
        angles_matrix = Matrix([[np.cos(theta_a)**2, np.sin(theta_a)**2, np.cos(theta_a) + np.sin(theta_a)],
                                [np.cos(theta_b)**2, np.sin(theta_b)**2, np.cos(theta_b) + np.sin(theta_b)],
                                [np.cos(theta_c)**2, np.sin(theta_c)**2, np.cos(theta_c) + np.sin(theta_c)]])
        solutions = defaultdict(dict)
        dico_length = len(epsilons_abc["epsilon_a"])
        for index in range(dico_length):
            epsilons_abc_matrix = Matrix([epsilons_abc["epsilon_a"][index], epsilons_abc["epsilon_b"][index], epsilons_abc["epsilon_c"][index]])
            equation = Eq(angles_matrix * epsilons_xy_matrix, epsilons_abc_matrix)
            solution = solve(equation, (eps_x, eps_y, gam_xy))
            solutions[index] = {key: round(value, 4) for key, value in solution.items()}
        return dict(solutions)
