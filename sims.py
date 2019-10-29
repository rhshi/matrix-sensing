import numpy as np 
from sklearn.preprocessing import normalize
import argparse
from utils import Logger


class MatrixSensing(object):
    def __init__(self, d, r, eta, log_file):

        self.logger = Logger(log_file)

        m = 5 * d * r
        matrices = np.random.randn(m, d, d)
        matrices_T = np.transpose(matrices, (0, 2, 1))
        self.matrices = (matrices + matrices_T) / 2
        self.matrices_T = np.transpose(self.matrices, (0, 2, 1))

        U = np.random.randn(d, r)
        U_normalized = normalize(U, norm='l2', axis=0)

        self.X_star = np.matmul(U_normalized, U_normalized.T)

        self.eta = eta
        self.d = d
        self.r = r
        self.identity = np.identity(d)

    def M_t(self, U):
        X = np.matmul(U, U.T)
        matmul = np.matmul(self.matrices_T, X - self.X_star)
        inner_product = np.trace(matmul, axis1=1, axis2=2)
        prods = inner_product.reshape(-1, 1, 1) * self.matrices

        # total = np.zeros((self.d, self.d))
        # for A in self.matrices:
        #     X = np.matmul(U, U.T)
        #     matrix_product = np.matmul(A.T, X - self.X_star)
        #     total += np.linalg.norm(matrix_product) * A

        return np.mean(prods, axis=0)

    def step(self, U):
        return np.matmul(self.identity - self.eta * self.M_t(U), U)

    def train_error(self, U):
        X = np.matmul(U, U.T)
        inner_product = np.trace(np.matmul(self.matrices_T, X), axis1=1, axis2=2)
        b = np.trace(np.matmul(self.matrices_T, self.X_star), axis1=1, axis2=2)
        numerator = np.sum((inner_product - b) ** 2)
        denominator = np.sum(b ** 2)

        return np.sqrt(numerator / denominator)

    def test_error(self, U):
        X = np.matmul(U, U.T)
        return np.linalg.norm(X - self.X_star) / np.linalg.norm(self.X_star)

    def go(self, alpha, iters, log_freq):

        self.logger.log("Parameters:")
        self.logger.log("dimension: {}".format(self.d))
        self.logger.log("rank: {}".format(self.r))
        self.logger.log("# sensing matrices: {}".format(5 * self.d * self.r))
        self.logger.log("lr: {}".format(self.eta))
        self.logger.log("alpha: {}".format(alpha))
        self.logger.log("iters: {}".format(iters))
        self.logger.log("----------------------")

        U = alpha * self.identity
        for i in range(iters):
            if i % log_freq == 0:
                self.logger.log("Iteration: {}".format(i))
                self.logger.log("Train error: {}".format(self.train_error(U)))
                self.logger.log("Test error: {}".format(self.test_error(U)))
                self.logger.log("----------------------")
            U = self.step(U)

        self.logger.log("----------------------")
        self.logger.log("Final")
        self.logger.log("Train error: {}".format(self.train_error(U)))
        self.logger.log("Test error: {}".format(self.test_error(U)))
        self.logger.log("----------------------")
        self.logger.log("")

        return U


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('-d', type=int, default=100)
    parser.add_argument('-r', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.0025)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=int(1e4))
    parser.add_argument('--log_freq', type=int, default=250)

    args = parser.parse_args()

    sim = MatrixSensing(args.d, args.r, args.eta, args.log_file)
    U = sim.go(args.alpha, args.iters, args.log_freq)

    return


if __name__ == "__main__":
    main()
