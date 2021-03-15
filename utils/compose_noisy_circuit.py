from typing import List, Tuple
import numpy as np
from random import random
from utils.error_channels import OneQubitErrorChannel
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate

def compose_noisy_circuit(circuit: QuantumCircuit, noise: List[Tuple[float, np.matrix]]):
    """Insert stochastic noisy operation to input QuantumCircuit
    This function and qiskit gate level simulation only treat unitary errors.

    """
    qr = QuantumRegister(1)
    noisy_circuit = QuantumCircuit(qr)
    for block in circuit:
        if block[0].name == 'measure':
            break
        # insert noise with montecarlo method
        r = random()
        prob_sum = 0
        for prob, noise_op in noise:
            prob_sum += prob
            if prob_sum > r:
                break
        op = np.dot(block[0].to_matrix(), noise_op)
        # noisy_circuit.append(block[0], [qr[0]])
        # noisy_circuit.append(UnitaryGate(noise_op), [qr[0]])
        noisy_circuit.append(UnitaryGate(op), [qr[0]])
    noisy_circuit.measure_all()
    return noisy_circuit