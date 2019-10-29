import numpy as np 
from sklearn.preprocessing import normalize
import argparse


class MatrixSensing(object):
    def __init__(self, d, r, eta):
        m = 5 * d * r
        self.matrices = np.random.randn(m, d, d)
        self.matrices_T = np.transpose(self.matrices, (0, 2, 1))

        U = np.random.randn(d, r)
        U_normalized = normalize(U, norm='l2', axis=0)

        self.X_star = np.matmul(U_normalized, U_normalized.T)

        self.eta = eta
        self.d = d
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
        numerator = 0
        denominator = 0
        for A in self.matrices:
            b = np.linalg.norm(np.matmul(A.T, self.X_star))
            X = np.matmul(U, U.T)
            numerator += (np.linalg.norm(np.matmul(A.T, X)) - b) ** 2
            denominator += b ** 2

        return np.sqrt(numerator / denominator)

    def test_error(self, U):
        X = np.matmul(U, U.T)
        return np.linalg.norm(X - self.X_star) / np.linalg.norm(self.X_star)

    def go(self, alpha, iters):
        U = alpha * self.identity
        for i in range(iters):
            print(i)
            U = self.step(U)

        return U


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, default=100)
    parser.add_argument('-r', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.0025)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iters', type=float, default=int(10e4))

    args = parser.parse_args()

    sim = MatrixSensing(args.d, args.r, args.eta)
    U = sim.go(args.alpha, args.iters)

    print("Train error: {}".format(sim.train_error(U)))
    print("Test error: {}".format(sim.test_error(U)))

    return


if __name__ == "__main__":
    main()