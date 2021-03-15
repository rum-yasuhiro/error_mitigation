from typing import List, Tuple, Union
import numpy as np
import sympy
from random import random

from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer
from qiskit.compiler import assemble
from qiskit.quantum_info import pauli_group
from qiskit.extensions import UnitaryGate



class OneQubitProbabilisticErrorCancellation: 
    """Probabilistic Error Cancellation for one qubit system
    
    """

    def __init__(self, kraus: list):
        """
        Args: 
            kraus: The list of Kraus operator
        """
        self.number_of_qubits = 1
        self.pauli_matrices = self.pauli_matrix()
        self.d = 2 ** self.number_of_qubits
        self.I = self.pauli_matrices[0]
        self.X = self.pauli_matrices[1]
        self.Y = self.pauli_matrices[2]
        self.Z = self.pauli_matrices[3]

        self.ptm_basis_op = self.prepare_basis_ptm()

        self.kraus = kraus

    def simulate_expected_value(self, experiments: List[Tuple[QuantumCircuit, int]] = None, cost_tot=None) -> list:
        """Simulate the expectec value of the probabilistic error cancelled quantum circuit

        Args: 
            experments: QuantumCircuit and parity
            cost_tot: The total cost of Error Mitigation
        Return: 
            list: The list of expected value of experiments
        """
        simulator = Aer.get_backend('qasm_simulator')
        if experiments is None: 
            experiments = self.experiments

        if cost_tot is None: 
            cost_tot = self.cost_tot

        counts = {'0': 0, '1': 0}
        for qc, parity in experiments:
            qobj = assemble(qc)
            job = simulator.run(qobj, shots=1)
            _counts= job.result().get_counts()
            if parity==1:
                counts['0'] += _counts.get('0', 0)
                counts['1'] += _counts.get('1', 0)
            else:
                counts['0'] += _counts.get('1', 0)
                counts['1'] += _counts.get('0', 0)
        e = self.expectation_value(counts) * cost_tot
        print("Expectation Value: ", e)
        self.ev = float(e)
        return self.ev 

    def pec_circuits(self, circuits: Union[List[QuantumCircuit], QuantumCircuit], num_trial: int = None) -> List[QuantumCircuit]:
        """Modify QuantumCircuit with pec recover operations

        Args:
            circuit: Input QuantumCircuit to be modified
            num_trial: number of trial of Probability error cancellation
        
        Return: 
            QuantumCircuit: QuantumCircuit inserted recover operations 
        """
        cost, probs = self.quasi_probability_method()

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits] * num_trial
        self.cost_tot = cost ** circuits[0].depth()
        self.experiments = [self.insert_recovers(circuit, probs) for circuit in circuits]
        return self.experiments, self.cost_tot


    def insert_recovers(self, circuit:QuantumCircuit, probs:dict)-> QuantumCircuit:
        """Create QuantumCircuit inserted recover operator"""
        
        qr = QuantumRegister(1)
        recoverd_qc = QuantumCircuit(qr)
        parity = 1
        for node in circuit:
            if node[0].name == 'measure' or node[0].name == 'barrier':
                break
            gate = node[0]  
            recover_gate, parity = self.monte_carlo_gate_selection(probs, parity)
            recoverd_qc.append(gate, [qr[0]])
            recoverd_qc.append(recover_gate, [qr[0]])
        recoverd_qc.measure_all()
        return recoverd_qc, parity

    def monte_carlo_gate_selection(self, probability_distribution: dict, parity: int) -> (UnitaryGate, int):
        r = random()
        pd = probability_distribution
        sum_prob = 0
        for label, _prob in pd.items():
            sum_prob += _prob
            if r<sum_prob:
                if label != 'iden':
                    parity *= -1
                recover_op = UnitaryGate(self.basis_ops[label])
                return recover_op, parity
        

    def prepare_basis_ptm(self):
        """Prepare pauli transfar matrices of 16 basis operations

        https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031027 
        のTable 1 についてPauli transfer matrixを計算
        """

        rx = 1/np.sqrt(2) * (self.I + 1J*self.X)
        ry = 1/np.sqrt(2) * (self.I + 1J*self.Y)
        rz = 1/np.sqrt(2) * (self.I + 1J*self.Z)
        ryz = 1/np.sqrt(2) * (self.Y + self.Z)
        rzx = 1/np.sqrt(2) * (self.Z + self.X)
        rxy = 1/np.sqrt(2) * (self.X + self.Y)
        pix = 1/2 * (self.I + self.X)
        piy = 1/2 * (self.I + self.Y)
        piz = 1/2 * (self.I + self.Z)
        piyz = 1/2 * (self.Y + 1j*self.Z)
        pizx = 1/2 * (self.Z+ 1j*self.X)
        pixy = 1/2 * (self.X+ 1j*self.Y)
        self.basis_ops = {
            "iden": self.I,
            "sigmaX": self.X,
            "sigmaY": self.Y,
            "sigmaZ": self.Z,
            "Rx": rx, 
            "Ry": ry,
            "Rz": rz,
            "Ryz": ryz,
            "Rzx": rzx,
            "Rxy": rxy,
            "PIx": pix,
            "PIy": piy,
            "PIz": piz,
            "PIyz": piyz,
            "PIzx": pizx,
            "PIxy": pixy,
        }

        return [self.ptm(_basisop, self.pauli_matrices, self.d) for _basisop in self.basis_ops.values()]
    
    def pauli_matrix(self):
        return [np.matrix(pauli.to_matrix()) for pauli in pauli_group(self.number_of_qubits)]

    def ptm(self, kraus, pauli_set, d):
        """Pauli Transfar matrix
        
        Args:
            kraus: List of Kraus operator as numpy.matrix
            pauli_set: pauli operators corresponds to dimension of system
            d: dimension of system
        """
        
        if not isinstance(kraus, list): 
            kraus = [kraus]
        def emap(K, rho):
            e_rho = np.array([[0.0, 0.0], [0.0, 0.0]])
            for K_i in K: 
                krhokdag = np.dot(K_i, np.dot(rho, K_i.getH()))
                e_rho = e_rho + krhokdag
            return e_rho

        mat = []
        for P_i in pauli_set:
            row = []
            for P_j in pauli_set:
                E_ij = 1/2 * np.trace(np.dot(P_i, emap(kraus, P_j)))
                row.append(E_ij)
            mat.append(row)
        return np.array(mat)

    def quasi_probability_method(self) -> (float, dict): 
        # calculate Pauli transfar matrix E and get inverse matrix
        E = self.ptm(self.kraus, self.pauli_matrices, self.d)
        E_inv = np.linalg.inv(E)

        # solve simultaneous equation to get recover probability from quasi-probability
        q_iden = sympy.Symbol('q_iden')
        q_sigmaX = sympy.Symbol('q_sigmaX')
        q_sigmaY = sympy.Symbol('q_sigmaY')
        q_sigmaZ = sympy.Symbol('q_sigmaZ')
        q_Rx = sympy.Symbol('q_Rx')
        q_Ry = sympy.Symbol('q_Ry')
        q_Rz = sympy.Symbol('q_Rz')
        q_Ryz = sympy.Symbol('q_Ryz')
        q_Rzx = sympy.Symbol('q_ Rzx')
        q_Rxy = sympy.Symbol('q_Rxy')
        q_PIx = sympy.Symbol('q_PIx')
        q_PIy = sympy.Symbol('q_PIy')
        q_PIz = sympy.Symbol('q_PIz')
        q_PIyz = sympy.Symbol('q_PIyz')
        q_PIzx = sympy.Symbol('q_PIzx')
        q_PIxy = sympy.Symbol('q_PIxy')
        quasi_probs = [q_iden, q_sigmaX, q_sigmaY, q_sigmaZ, q_Rx, q_Ry, q_Rz, q_Ryz, q_Rzx, q_Rxy, q_PIx, q_PIy, q_PIz, q_PIyz, q_PIzx, q_PIxy]

        solutions = self.solve_compose_simulequ(quasi_probs, E_inv)

        # get cost of PEC and recover probability
        self.cost = self.qem_cost(solutions)

        recover_probabilities = {
            "iden": self.quasi_to_prob(solutions[q_iden], self.cost),
            "sigmaX": self.quasi_to_prob(solutions[q_sigmaX], self.cost),
            "sigmaY": self.quasi_to_prob(solutions[q_sigmaY], self.cost),
            "sigmaZ": self.quasi_to_prob(solutions[q_sigmaZ], self.cost),
            "Rx": self.quasi_to_prob(solutions[q_Rx], self.cost), 
            "Ry": self.quasi_to_prob(solutions[q_Ry], self.cost),
            "Rz": self.quasi_to_prob(solutions[q_Rz], self.cost),
            "Ryz": self.quasi_to_prob(solutions[q_Ryz], self.cost),
            "Rzx": self.quasi_to_prob(solutions[q_Rzx], self.cost),
            "Rxy": self.quasi_to_prob(solutions[q_Rxy], self.cost),
            "PIx": self.quasi_to_prob(solutions[q_PIx], self.cost),
            "PIy": self.quasi_to_prob(solutions[q_PIy], self.cost),
            "PIz": self.quasi_to_prob(solutions[q_PIz], self.cost),
            "PIyz": self.quasi_to_prob(solutions[q_PIyz], self.cost),
            "PIzx": self.quasi_to_prob(solutions[q_PIzx], self.cost),
            "PIxy": self.quasi_to_prob(solutions[q_PIxy], self.cost),
        }

        return self.cost, recover_probabilities
    
    def solve_compose_simulequ(self, quasi_probabilities, E_inv):
        equ_list = []
        for i in range(4): 
            for j in range(4):
                equ_ij = 0
                for l, basis_op in enumerate(self.ptm_basis_op):
                    equ_ij += quasi_probabilities[l]*basis_op[i][j]
                equ_list.append(equ_ij - E_inv[i][j])

        return sympy.solve(equ_list)

    # cost of QEM (gamma)
    def qem_cost(self, solutions):
        values = list(solutions.values())
        cost = 0
        for q_i in values:
            cost += abs(q_i)
        print("cost:", cost)
        return cost

    # Quasi-probability to probability of recover insertion
    def quasi_to_prob(self, q, cost):
        p = abs(q)/cost
        return p 

    def expectation_value(self, probability_distribution):
        gra_pop = probability_distribution.get('0', 0)
        exc_pop = probability_distribution.get('1', 0)
        return (gra_pop*1 + exc_pop*(-1)) / (gra_pop+exc_pop)
class NumberOfQubitError(Exception):
    pass