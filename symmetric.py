import numpy as np
from sims import SymmetricMS


class SymmetricEmpirical(SymmetricMS):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self, U):
        return np.matmul(self.identity - self.eta * self.M_t(U, U), U)


class SymmetricPopulation(SymmetricMS):
    def __init__(self, *args):
        super().__init__(*args)

    def step(self, U):
        diff = np.matmul(U, U.T) - self.X_star
        return U - 2 * self.eta * np.matmul(diff, U)
