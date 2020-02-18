# import argparse
import numpy as np
from sklearn.preprocessing import normalize
from base import MatrixSensing


class SymmetricMS(MatrixSensing):

    def __init__(self, *args):
        super().__init__(*args)

        u = np.random.randn(self.d, self.r)
        self.u_star = normalize(u, norm='l2', axis=0)

        self.X_star = np.matmul(self.u_star, self.u_star.T)

    def step(self):
        raise NotImplementedError

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
                if self.verbose:
                    rew = np.squeeze(np.matmul(U.T, self.u_star))
                    err = np.matmul(self.identity - self.X_star, U) 
                    assert np.allclose(np.outer(self.u_star, rew) + err, U)
                    self.logger.log("Reward: {}".format(np.linalg.norm(rew)))
                    self.logger.log("Error: {}".format(np.linalg.norm(err) ** 2))
                self.logger.log("----------------------")
            U = self.step(U)

        self.logger.log("----------------------")
        self.logger.log("Final")
        self.logger.log("Train error: {}".format(self.train_error(U, U)))
        self.logger.log("Test error: {}".format(self.test_error(U, U)))
        if self.verbose:
            rew = np.squeeze(np.matmul(U.T, self.u_star))
            err = np.matmul(self.identity - self.X_star, U) 
            assert np.allclose(np.outer(self.u_star, rew) + err, U)
            self.logger.log("Reward: {}".format(np.linalg.norm(rew)))
            self.logger.log("Error: {}".format(np.linalg.norm(err) ** 2))
        self.logger.log("----------------------")
        self.logger.log("")

        return U


class AsymmetricMS(MatrixSensing):

    def __init__(self, *args):
        super().__init__(*args)

        u = np.random.randn(self.d, self.r)
        self.u_star = normalize(u, norm='l2', axis=0)

        v = np.random.randn(self.d, self.r)
        self.v_star = normalize(v, norm='l2', axis=0)

        self.X_star = np.matmul(self.u_star, self.v_star.T)

    def step(self):
        raise NotImplementedError

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
                self.log(U, V)
            U, V = self.step(U, V)
        self.logger.log("----------------------")
        self.logger.log("Final")
        self.log(U, V)
        self.logger.log("")

        return U, V

    def log(self, U, V):
        self.logger.log("Train error: {}".format(self.train_error(U, V)))
        self.logger.log("Test error: {}".format(self.test_error(U, V)))
        if self.verbose:
            M = np.matmul(U, V) - self.X_star
            id_u = np.matmul(self.u_star, self.u_star.T)
            id_v = np.matmul(self.v_star, self.v_star.T)
            rew_u = np.squeeze(np.matmul(U.T, self.u_star))
            rew_v = np.squeeze(np.matmul(V.T, self.v_star))
            err_u = np.matmul(self.identity - id_u, U) 
            err_v = np.matmul(self.identity - id_v, V)
            assert np.allclose(np.outer(self.u_star, rew_u) + err_u, U)
            assert np.allclose(np.outer(self.v_star, rew_v) + err_v, V)
            self.logger.log("Reward U: {}".format(np.linalg.norm(rew_u)))
            self.logger.log("Error U: {}".format(np.linalg.norm(err_u) ** 2))
            self.logger.log("Reward V: {}".format(np.linalg.norm(rew_v)))
            self.logger.log("Error V: {}".format(np.linalg.norm(err_v) ** 2))   
            if self.r == 1:
                self.logger.log("Inner: {}".format(np.dot(rew_u, rew_v)))
        self.logger.log("----------------------")

        return
