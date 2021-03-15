import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from probabilistic_error_cancellation import OneQubitProbabilisticErrorCancellation
from utils import OneQubitErrorChannel
from utils.compose_noisy_circuit import compose_noisy_circuit


def evaluate_dephasing_error(depth, num_shots, num_trial, save_path):
    # load and get provider from IBM Q api
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')

    # get IBMQ backend
    backend = provider.get_backend('ibmqx2')
    
    # prepare quantum circuit
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    qc.h(qr[0])
    for _ in range(depth):
        qc.z(qr[0])
    qc.h(qr[0])
    qc.measure(qr, cr)

    # get t2 time and sx gate time of IBMQ QX 2 qubit 0
    t2 = backend.properties().t2(0)
    gt = backend.properties()._gates['sx'][(0,)]['gate_length'][0]

    # compose noisy circuit
    cptp_map =OneQubitErrorChannel()
    kraus = cptp_map.phase_damping_channel(t2, gt)
    noise = cptp_map.phase_damping_noise(t2, gt)


    # Probabilistic Error Cancellation
    pec = OneQubitProbabilisticErrorCancellation(kraus)

    print('######################## Expected values ########################')
    noisy_ev = []
    pec_ev = []
    for _ in range(num_trial):
        # noisy_circuit_set
        noisy_circuits = [compose_noisy_circuit(qc, noise) for _ in range(num_shots)]
        noisy_experiments = [(noisy_qc, 1) for noisy_qc in noisy_circuits]
        _ev = pec.simulate_expected_value(experiments=noisy_experiments, cost_tot=1)
        noisy_ev.append(_ev)

        # pec
        pec_experiments, cost_tot = pec.pec_circuits(noisy_circuits)
        _ev = pec.simulate_expected_value(experiments=pec_experiments, cost_tot=cost_tot)
        pec_ev.append(_ev)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if depth%2 == 0:
        e_noiseless = [1] * num_trial
    else:
        e_noiseless = [-1] * num_trial
    ax.hist(e_noiseless, color='c', edgecolor='k', alpha=0.5, bins=30, label='Noiseless')
    ax.hist(noisy_ev, color='m', edgecolor='k', alpha=0.5, bins=10, label="Noisy")
    ax.hist(pec_ev, color='g', edgecolor='k', alpha=0.5, bins=30, label="PEC")


    plt.legend()
    plt.show()
    plt.savefig(save_path)

if __name__ == "__main__":
    depth = 101
    num_shots = 1000
    num_trial = 1000
    save_path = '/Users/rum/Documents/error_mitigation/dphasing.png'
    evaluate_dephasing_error(depth, num_shots, num_trial, save_path)