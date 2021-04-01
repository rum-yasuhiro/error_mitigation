import numpy as np
from qiskit.quantum_info import pauli_group


class OneQubitErrorChannel():
    def __init__(self):
        self.d = 2
        self.pauli_matrix = pauli_group(1)
        self.I = np.matrix(self.pauli_matrix[0].to_matrix())
        self.X = np.matrix(self.pauli_matrix[1].to_matrix())
        self.Y = np.matrix(self.pauli_matrix[2].to_matrix())
        self.Z = np.matrix(self.pauli_matrix[3].to_matrix())

    def depolarizing_channel(self, p):
        K_0 = np.sqrt((1 - 3*p/4)) * self.I
        K_1 = np.sqrt(p/4) * self.X
        K_2 = np.sqrt(p/4) * self.Y
        K_3 = np.sqrt(p/4) * self.Z
        depolarize = [K_0, K_1, K_2, K_3]
        return depolarize

    def depolarizing_noise(self, p):
        return [(1 - 3*p/4, self.I), (p/4, self.X), (p/4, self.Y), (p/4, self.Z)]

    def phase_damping_channel(self, t2, dt):
        p = (1 - np.e**(-2*dt*(1/t2))) / 2
        K_0 = np.sqrt((1 - p)) * self.I
        K_1 = np.sqrt(p) * self.Z
        phase_damping = [K_0, K_1]
        return phase_damping

    def phase_damping_noise(self, t2, dt):
        p = (1 - np.e**(-2*dt*(1/t2))) / 2
        return [(1 - p, self.I), (p, self.Z)]

    def amplitude_damping_channel(self, t1, dt):
        gamma_t = 1 - np.e**((-1/t1)*dt)
        K_0 = np.matrix([[1, 0], [0, np.sqrt(1-gamma_t)]])
        K_1 = np.matrix([[0, np.sqrt(gamma_t)], [0, 0]])
        amplitude_damping = [K_0, K_1]
        return amplitude_damping

    def pauli_matrix(self):
        return [np.matrix(pauli.to_matrix()) for pauli in pauli_group(1)]
