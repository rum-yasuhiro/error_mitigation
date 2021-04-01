import numpy as np

from error_mitigation.probabilistic_error_cancellation.one_qubit_pec import OneQubitProbabilisticErrorCancellation
from error_mitigation.utils.error_channels import OneQubitErrorChannel


class TestOneQubitProbabilisticErrorCancellation:

    def test_simulate_unital_noise(self, prepare_hzh_qc):
        # prepare input QuantumCircuit
        qc = prepare_hzh_qc

        # prepare error channel
        oneQerr = OneQubitErrorChannel()
        p = 0.005
        depol = oneQerr.depolarizing_channel(p)

        # simulate
        smpl = []
        for _ in range(10):
            # define PEC circuit and calculate its cost
            pec = OneQubitProbabilisticErrorCancellation(depol)
            rcv_qcs, cost_tot = pec.pec_circuits(circuits=qc, num_trial=100)

            assert len(rcv_qcs) == 100

            rcv_ev = pec.simulate_unital_noise()
            smpl.append(rcv_ev)
        assert np.mean(smpl) > 0.9 and np.mean(smpl) < 1.1

    def test_simulate_nonunital_noise(self, prepare_hzh_qc, prepare_ibmqx2):
        # prepare input QuantumCircuit
        qc = prepare_hzh_qc

        # get your backend
        ibmqx2 = prepare_ibmqx2
        T1 = ibmqx2.properties().t1(0)
        gate_time = ibmqx2.properties()._gates['sx'][(0,)]['gate_length'][0]

        # prepare error channel
        oneQerr = OneQubitErrorChannel()
        amp = oneQerr.amplitude_damping_channel(T1, gate_time)

        # simulatete
        smpl = []
        for _ in range(100):
            # define PEC circuit and calculate its cost
            pec = OneQubitProbabilisticErrorCancellation(amp)
            rcv_qcs, cost_tot = pec.pec_circuits(circuits=qc, num_trial=100)

            assert len(rcv_qcs) == 100

            rcv_ev = pec.simulate_nonunital_noise()
            smpl.append(rcv_ev)
        print(np.mean(smpl))
