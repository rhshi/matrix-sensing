import argparse
import numpy as np
from sklearn.preprocessing import normalize
from base import MatrixSensing


class SymmetricMS(MatrixSensing):

    def __init__(self, d, r, eta, log_file):
        super().__init__(d, r, eta, log_file)

        U = np.random.randn(d, r)
        U_normalized = normalize(U, norm='l2', axis=0)

        self.X_star = np.matmul(U_normalized, U_normalized.T)

    def step(self, U):
        return np.matmul(self.identity - self.eta * self.M_t(U, U), U)


    def go(self, alpha, iters, log_freq):
        self.logger.log_init(
            self.d,
            self.r,
            self.eta,
            alpha,
            iters
        )

        U = alpha * (self.identity + np.random.randn(self.d, self.d))
        for i in range(iters):
            if i % log_freq == 0:
                self.logger.log("Iteration: {}".format(i))
                self.logger.log("Train error: {}".format(self.train_error(U, U)))
                self.logger.log("Test error: {}".format(self.test_error(U, U)))
                self.logger.log("----------------------")
            U = self.step(U)

        self.logger.log("----------------------")
        self.logger.log("Final")
        self.logger.log("Train error: {}".format(self.train_error(U, U)))
        self.logger.log("Test error: {}".format(self.test_error(U, U)))
        self.logger.log("----------------------")
        self.logger.log("")

        return U


class AsymmetricMS(MatrixSensing):

    def __init__(self, d, r, eta, log_file):
        super().__init__(d, r, eta, log_file)

        U = np.random.randn(d, r)
        U_normalized = normalize(U, norm='l2', axis=0)

        V = np.random.randn(d, r)
        V_normalized = normalize(V, norm='l2', axis=0)

        self.X_star = np.matmul(U_normalized, V_normalized.T)

    def step(self, U, V):
        M_t = self.M_t(U, V)
        new_U = U - self.eta * np.matmul(M_t, V)
        new_V = V - self.eta * np.matmul(M_t, U)
        return new_U, new_V

    def go(self, alpha, iters, log_freq):
        self.logger.log_init(
            self.d,
            self.r,
            self.eta,
            alpha,
            iters
        )

        U = alpha * (self.identity + np.random.randn(self.d, self.d))
        V = alpha * (self.identity + np.random.randn(self.d, self.d))
        for i in range(iters):
            if i % log_freq == 0:
                self.logger.log("Iteration: {}".format(i))
                self.logger.log("Train error: {}".format(self.train_error(U, V)))
                self.logger.log("Test error: {}".format(self.test_error(U, V)))
                self.logger.log("----------------------")
            U, V = self.step(U, V)

        self.logger.log("----------------------")
        self.logger.log("Final")
        self.logger.log("Train error: {}".format(self.train_error(U, V)))
        self.logger.log("Test error: {}".format(self.test_error(U, V)))
        self.logger.log("----------------------")
        self.logger.log("")

        return U


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=str)
    parser.add_argument('--mode', tpe=str, default="sym")
    parser.add_argument('-d', type=int, default=100)
    parser.add_argument('-r', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.0025)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=int(1e4))
    parser.add_argument('--log_freq', type=int, default=250)

    args = parser.parse_args()

    if args.mode == "sym":
        sim = SymmetricMS(args.d, args.r, args.eta, args.log_file)
    elif args.mode == "asym":
        sim = AsymmetricMS(args.d, args.r, args.eta, args.log_file)
    
    U = sim.go(args.alpha, args.iters, args.log_freq)

    return


if __name__ == "__main__":
    main()