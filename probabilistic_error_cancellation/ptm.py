from typing import List, Union
import numpy as np

def ptm(kraus, pauli_set, d):
    """Pauli Transfar matrix
    
    Args:
        kraus: List of Kraus operator as numpy.matrix
        pauli_set: pauli operators corresponds to dimension of system
        d: dimension of system
    """
    
    if not isinstance(U, list): 
        U = [U]
    def emap(E, rho):
        e_rho = np.array([[0.0, 0.0], [0.0, 0.0]])
        for E_i in E: 
            erhoe = np.dot(E_i, np.dot(rho, E_i.getH()))
            e_rho = e_rho + erhoe
        return e_rho

    mat = []
    for P_i in pauli_set:
        row = []
        for P_j in pauli_set:
            E_ij = 1/2 * np.trace(np.dot(P_i, emap(U, P_j)))
            row.append(E_ij)
        mat.append(row)
    return np.array(mat)