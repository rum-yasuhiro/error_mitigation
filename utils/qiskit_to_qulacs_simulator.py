import time
from collections import Counter

import numpy as np
import qulacs
from qiskit import QuantumCircuit


def get_noise_gate_list(target_index_list: list, noise_map_list: list) -> list:
    """Get noise gate list from list of CPTP maps

    Generate noise gate list for qulacs circuit
    In the case of probabilistic X and Z noise on 0-th and 2-nd qubits,
    target_index_list = [0,2]
    noise_map_list = [noise_X, noise_Z]
    where
    noise_X = [sqrt(1-px)*I, sqrt(px)*X]
    noise_Z = [sqrt(1-pz)*I, sqrt(pz)*Z]
    I = [[1,0], [0,1]]
    X = [[0,1], [1,0]]
    Z = [[1,0], [0,-1]]

    Args:
        target_index_list (list): List of target indices of noisy gates
        noise_map_list (list): List of CPTP maps. Each CPTP map has list of Kraus operators as matrix.

    Returns:
        list: list of noise gates
    """
    noise_gate_list = []
    for kraus_mat_list in noise_map_list:
        for target_index in target_index_list:
            kraus_op_list = []
            for kraus_mat in kraus_mat_list:
                kraus_op = qulacs.gate.DenseMatrix([target_index], kraus_mat)
                kraus_op_list.append(kraus_op)
            noise = qulacs.gate.CPTP(kraus_op_list)
            noise_gate_list.append(noise)
        # return noise_gate_list
    return noise_gate_list


def simulate_with_qulacs(num_qubit: int, qiskit_circuit: QuantumCircuit, num_shots: int, noise_model: list = None) -> dict:
    """Simulate qiskit's quantum circuit with qulacs

    Simulate qunatum circuit defined by qiskit with qulacs.
    We can apply noise to the quantum circuit.
    If noise_model is None, quantum circuit is assumed to be noiseless.

    We assume quantum circuit is finally measured by Z-basis.
    Thus, the "measure" gate is ignored.

    If we need to simulate noisy quantum circuit, noise model must be a list of which the length is equal to that of qiskit_circuit.
    The i-th element of noise_model contains noise_map_list of the i-th gate.
    See get_noise_gate_list function for the definition of noise_map_list.

    Args:
        num_qubit (int): number of qubits
        qiskit_circuit (QuantumCircuit): qiskit's quantum circuit
        num_shots (int): number of shots to measure
        noise_model (list, optional): noise model. Defaults to None.

    Raises:
        ValueError: information of noise model is invalid, or unknown gate name is contained in quantum circuit

    Returns:
        dict: list of shots
    """
    if noise_model is not None and len(noise_model) != len(qiskit_circuit):
        raise ValueError()
    qulacs_circuit = qulacs.QuantumCircuit(num_qubit)
    state = qulacs.DensityMatrix(num_qubit)

    for gate_index, gate in enumerate(qiskit_circuit):
        # get gate
        gate_name = gate[0].name
        gate_name_to_func = {
            "x": qulacs.gate.X,
            "y": qulacs.gate.Y,
            "z": qulacs.gate.Z,
            "h": qulacs.gate.H,
            "cx": qulacs.gate.CNOT,
            "id": qulacs.gate.Identity,
            "measure": None,
        }
        if gate_name not in gate_name_to_func:
            raise ValueError("gate {} is not implemented".format(gate_name))
        gate_func = gate_name_to_func[gate_name]

        # get index
        target_index_list = [target_qubit_info.index for target_qubit_info in gate[1]]

        # add gate
        if gate_func is not None:
            gate = gate_func(*target_index_list)
            gate.update_quantum_state(state)

        # add noise
        if noise_model is not None:
            noise_map_list = noise_model[gate_index]
            noise_gate_list = get_noise_gate_list(target_index_list, noise_map_list)
            for noise_gate in noise_gate_list:
                noise_gate.update_quantum_state(state)

    shot_result = state.sampling(num_shots)
    shot_dict = dict(Counter(shot_result))
    return shot_dict

