import numpy as np
from sims import SymmetricMS


class AsymmetricEmpirical(AsymmetricMS):
    def __init__(self, d, r, eta, log_file):
        super().__init__(d, r, eta, log_file)

    def step(self, U, V):
        M_t = self.M_t(U, V)
        new_U = U - self.eta * np.matmul(M_t, V)
        new_V = V - self.eta * np.matmul(M_t.T, U)
        return new_U, new_V

class AsymmetricPopulation(AsymmetricMS):
    def __init__(self, d, r, eta, log_file):
        super().__init__(d, r, eta, log_file)

    def step(self, U, V):
        diff = np.matmul(U, V.T) - self.X_star
        return U - 2 * self.eta * np.matmul(diff, U)