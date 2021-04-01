import pytest
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ


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


@pytest.fixture
def prepare_ibmqx2():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmqx2')
    return backend
