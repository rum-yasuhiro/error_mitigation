import numpy as np
from qiskit.quantum_info import pauli_group

class OneQubitErrorChannel(): 
    def __init__():
        self.d = 2
        self.pauli_group = self.pauli_matrix
        self.I = self.pauli_group[0]
        self.X = self.pauli_group[1]
        self.Y = self.pauli_group[2]
        self.Z = self.pauli_group[3]

    def depolarizing_channel(self, p): 
        K_0 = np.sqrt((1 - 3*p/4)) * self.I
        K_1 = np.sqrt(p/4) * self.X
        K_2 = np.sqrt(p/4) * self.Y
        K_3 = np.sqrt(p/4) * self.Z
        depolarize = [K_0, K_1, K_2, K_3]
        return depolarize
    
    def phase_damping_channel(self, t2, dt): 
        p = (1 - np.e**(-2*dt*(1/t2))) / 2
        K_0 = np.sqrt((1 - p)) * self.I
        K_1 = np.sqrt(p) * self.Z
        phase_damping = [K_0, K_1]
        return phase_damping
    
    def amplitude_damping_channel(self, t1, dt):
        gamma_t = 1 - np.e^(1/t1)*dt
        K_0 = np.matrix([[1, 0], [0, np.sqrt(1-gamma_t)]])
        K_1 = np.matrix([[0, np.sqrt(gamma_t)], [0, 0]])
        amplitude_damping = [K_0, K_1]
        return amplitude_damping

    def pauli_matrix(self):
        return [np.matrix(pauli.to_matrix()) for pauli in pauli_group(1)]