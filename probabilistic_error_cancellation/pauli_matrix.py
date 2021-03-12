from qiskit.quantum_info import pauli_group
from numpy import matrix

def pauli_matrix(number_of_qubits)
    return [matrix(pauli.to_matrix()) for pauli in pauli_group(number_of_qubits)]
    