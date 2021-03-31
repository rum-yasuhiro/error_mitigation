import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


@pytest.fixture
def prepare_hzh_qc():
    # prepare QuantumCircuit used in test
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    for _ in range(10):  # expectation value of Z meas will be 1
        qc.z(qr[0])
    qc.h(qr[0])
    qc.measure(qr, cr)

    return qc
