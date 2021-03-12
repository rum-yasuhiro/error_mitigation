import numpy as np
from probabilistic_error_cancellation import ptm

class ProbabilisticErrorCancellation():

    def __init__(self , U, P_list, d): 
        self.I = np.matrix([[1, 0], [0, 1]])
        self.X = np.matrix([[0, 1], [1, 0]])
        self.Y = np.matrix([[0, 1j], [-1j, 0]])
        self.Z = np.matrix([[1, 0], [0, -1]])
        self.P_list = P_list
        self.d = d

        self.E_I = ptm(self.I, P, d)
        self.E_X = ptm(self.X, P, d)
        self.E_Y = ptm(self.Y, P, d)
        self.E_Z = ptm(self.Z, P, d)
        self.E_Rx = ptm(1/np.sqrt(2) * (I + 1J*X), P, d)

