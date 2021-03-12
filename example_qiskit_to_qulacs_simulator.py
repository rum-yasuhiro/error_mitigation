from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def build_circuit(num_qubit: int, num_depth: int) -> QuantumCircuit:
    """Build test circuit with qiskit

    Args:
        num_qubit (int): number of qubit
        num_depht (int): number of depth

    Returns:
        QuantumCircuit: qiskit's qunatum circuit
    """
    num_qreg = num_qubit
    num_creg = num_qubit
    qreg = QuantumRegister(num_qreg)
    creg = ClassicalRegister(num_creg)
    circuit = QuantumCircuit(qreg, creg)
    
    circuit.h(0)
    for _ in range(num_depth):
        circuit.z(0)
    circuit.h(0)
    circuit.measure(qreg, creg)
    return circuit


def get_dephase_noise(error_prob: float) -> list:
    """Get noise map of dephasing noise

    Args:
        error_prob (float): error probability

    Returns:
        list: list of kraus operators
    """
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    dephase = [np.sqrt(1 - error_prob) * I, np.sqrt(error_prob) * Z]
    return dephase


def get_bitflip_noise(error_prob: float) -> list:
    """Get noise map of bitflip noise

    Args:
        error_prob (float): error probability

    Returns:
        list: list of kraus operators
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    dephase = [np.sqrt(1 - error_prob) * I, np.sqrt(error_prob) * X]
    return dephase



# create circuit
num_qubit = 3
num_depth = 3
qiskit_circuit = build_circuit(num_qubit, num_depth)
num_gate = len(qiskit_circuit)
print(qiskit_circuit)

# simulate without noise
num_shot = 100000
result = simulate_with_qulacs(num_qubit, qiskit_circuit, num_shot)
print("clean", result)

# simulate with noise
def get_noise_model(error_prob: float, num_gate: int) -> dict:
    # create noise model: i-th element is list of CPTP map. Each CPTP-map is a list of Kraus operators.
    noise_per_gate = [get_dephase_noise(error_prob), get_bitflip_noise(error_prob)]
    noise_model = [noise_per_gate] * num_gate
    return noise_model

for error_prob in [0.01, 0.1, 0.2, 0.5]:
    noise_model = get_noise_model(error_prob, num_gate)
    result = simulate_with_qulacs(num_qubit, qiskit_circuit, num_shot, noise_model)
    print("error_rate={}".format(error_prob), result)

