import numpy as np 
from utils import Logger


class MatrixSensing(object):
    def __init__(self, d, r, eta, log_file, verbose):

        self.logger = Logger(log_file)

        self.verbose = verbose

        m = 5 * d * r
        matrices = np.random.randn(m, d, d)
        matrices_T = np.transpose(matrices, (0, 2, 1))
        self.matrices = (matrices + matrices_T) / 2
        self.matrices_T = np.transpose(self.matrices, (0, 2, 1))

        self.eta = eta
        self.d = d
        self.r = r
        self.identity = np.identity(d)

    def M_t(self, U, V):
        X = np.matmul(U, V.T)
        matmul = np.matmul(self.matrices_T, X - self.X_star)
        inner_product = np.trace(matmul, axis1=1, axis2=2)
        prods = inner_product.reshape(-1, 1, 1) * self.matrices
        return np.mean(prods, axis=0)

    def train_error(self, U, V):
        X = np.matmul(U, V.T)
        inner_product = np.trace(np.matmul(self.matrices_T, X), axis1=1, axis2=2)
        b = np.trace(np.matmul(self.matrices_T, self.X_star), axis1=1, axis2=2)
        numerator = np.sum((inner_product - b) ** 2)
        denominator = np.sum(b ** 2)

        return np.sqrt(numerator / denominator)

    def test_error(self, U, V):
        X = np.matmul(U, V.T)
        return np.linalg.norm(X - self.X_star) / np.linalg.norm(self.X_star)

    def step(self):
        raise NotImplementedError

    def go(self):
        raise NotImplementedError

