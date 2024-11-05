from composite import Composite, CompositeType
from sympy import symbols
import numpy as np


class Laminate:
    def __init__(self, thetas: list, composite_type: CompositeType, h: float):
        """
        Laminate class that builds a laminate from a list of angles and a composite type
        :param thetas: list of angles for the laminate layers in degrees
        :param composite_type: CompositeType
        :param h: Thickness of the laminate in mm
        """
        self.thetas = thetas
        self.composite_type = composite_type
        self.h = h/1000
        self.composites = [Composite(angle=theta, composite_type=composite_type) for theta in thetas]
        self.abd_matrix_cache = None

    @property
    def q_k(self):
        q_k = None
        for composite in self.composites:
            if q_k is None:
                q_k = composite.global_q_matrix
            else:
                q_k += composite.global_q_matrix
        return q_k

    @property
    def abd_matrix(self):
        if self.abd_matrix_cache is not None:
            return self.abd_matrix_cache
        n = len(self.thetas)
        H = n * self.h
        z = np.zeros((n + 1))
        z[0] = -H / 2
        for i in range(1, len(z)):
            z[i] = z[i - 1] + self.h

        a_matrix = np.zeros((3, 3))
        b_matrix = np.zeros((3, 3))
        d_matrix = np.zeros((3, 3))

        for i in range(n):
            q_matrix = self.composites[i].global_q_matrix
            a_matrix = a_matrix + q_matrix * (z[i + 1] - z[i])
            b_matrix = b_matrix + q_matrix * (z[i + 1] ** 2 - z[i] ** 2) / 2
            d_matrix = d_matrix + q_matrix * (z[i + 1] ** 3 - z[i] ** 3) / 3
        self.abd_matrix_cache = a_matrix, b_matrix, d_matrix
        return a_matrix, b_matrix, d_matrix

    @property
    def a_matrix(self):
        return self.abd_matrix[0]

    @property
    def inv_a_matrix(self):
        return np.linalg.inv(self.a_matrix)

    @property
    def b_matrix(self):
        return self.abd_matrix[1]

    @property
    def inv_b_matrix(self):
        return np.linalg.inv(self.b_matrix)

    @property
    def d_matrix(self):
        return self.abd_matrix[2]

    @property
    def inv_d_matrix(self):
        return np.linalg.inv(self.d_matrix)
