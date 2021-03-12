import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import pauli_group



class OneQubitProbabilisticErrorCancellation: 

    def __init__(self, kraus):
            
        self.number_of_qubits = 1
        self.pauli_matrices = self.pauli_matrix()
        self.d = 2 ** self.number_of_qubits
        self.I = self.pauli_matrices[0]
        self.X = self.pauli_matrices[1]
        self.Y = self.pauli_matrices[2]
        self.Z = self.pauli_matrices[3]

        self.ptm_basis_op = self.prepare_basis_ptm()

        self.kraus = kraus

    def run(self, circuit: QuantumCircuit):
        
        qr_rec = QuantumRegister(1)
        recover_circuit = QuantumCircuit(qr_rec)

        
        return circuit 

    def insert_recovers(self, circuit, qreg):
        return circuit

        return recover_circuit

    def prepare_basis_ptm(self):
        """
        https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.031027 
        のTable 1 についてPauli transfer matrixを計算
        """

        rx = 1/np.sqrt(2) * (self.I + 1J*self.X)
        ry = 1/np.sqrt(2) * (self.I + 1J*self.Y)
        rz = 1/np.sqrt(2) * (self.I + 1J*self.Z)
        ryz = 1/np.sqrt(2) * (self.Y + self.Z)
        rzx = 1/np.sqrt(2) * (self.Z + self.X)
        rxy = 1/np.sqrt(2) * (self.X + self.Y)
        PIx = 1/2 * (self.I + self.X)
        PIy = 1/2 * (self.I + self.Y)
        PIz = 1/2 * (self.I + self.Z)
        PIyz = 1/2 * (self.Y + 1j*self.Z)
        PIzx = 1/2 * (self.Z+ 1j*self.X)
        PIxy = 1/2 * (self.X+ 1j*self.Y)
        _basis_ops = [self.I, self.X, self.Y, self.Z, rx, ry, rz, ryz, rzx, rxy, PIx, PIy, PIz, PIyz, PIzx, PIxy]

        return [self.ptm(_basisop, self.pauli_matrices, self.d) for _basisop in _basis_ops]
    
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
                erhoe = np.dot(K_i, np.dot(rho, K_i.getH()))
                e_rho = e_rho + erhoe
            return e_rho

        mat = []
        for P_i in pauli_set:
            row = []
            for P_j in pauli_set:
                E_ij = 1/2 * np.trace(np.dot(P_i, emap(kraus, P_j)))
                row.append(E_ij)
            mat.append(row)
        return np.array(mat)

    def quasi_probability_method(self): 
        # calculate Pauli transfar matrix E and get inverse matrix
        E = ptm(self.kraus, self.pauli_matrices, self.d)
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
        q_list = [q_iden, q_sigmaX, q_sigmaY, q_sigmaZ, q_Rx, q_Ry, q_Rz, q_Ryz, q_Rzx, q_Rxy, q_PIx, q_PIy, q_PIz, q_PIyz, q_PIzx, q_PIxy]

        solutions = solve_compose_simulequ(q_list, E_inv)

        # get cost of PEC and recover probability
        self.cost = self.qem_cost(solutions)

        self.recover_probabilities = {
            "p_iden": quasi_to_prob(solutions[q_iden], self.cost),
            "p_sigmaX": quasi_to_prob(solutions[q_sigmaX], self.cost),
            "p_sigmaY": quasi_to_prob(solutions[q_sigmaY], self.cost),
            "p_sigmaZ": quasi_to_prob(solutions[q_sigmaZ], self.cost),
            "p_Rx": quasi_to_prob(solutions[q_Rx], self.cost), 
            "p_Ry": quasi_to_prob(solutions[q_Ry], self.cost),
            "p_Rz": quasi_to_prob(solutions[q_Rz], self.cost),
            "p_Ryz": quasi_to_prob(solutions[q_Ryz], self.cost),
            "p_Rzx": quasi_to_prob(solutions[q_Rzx], self.cost),
            "p_Rxy": quasi_to_prob(solutions[q_Rxy], self.cost),
            "p_PIx": quasi_to_prob(solutions[q_PIx], self.cost),
            "p_PIy": quasi_to_prob(solutions[q_PIy], self.cost),
            "p_PIz": quasi_to_prob(solutions[q_PIz], self.cost),
            "p_PIyz": quasi_to_prob(solutions[q_PIyz], self.cost),
            "p_PIzx": quasi_to_prob(solutions[q_PIzx], self.cost),
            "p_PIxy": quasi_to_prob(solutions[q_PIxy], self.cost),
        }
    
    def solve_compose_simulequ(self, q_list, E_inv):
        equ_list = []
        for i in range(4): 
            for j in range(4):
                equ_ij = 0
                for l in range(16):
                    equ_ij += q_list[l]*Basis_op[l][i][j]
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


class NumberOfQubitError(Exception):
    pass