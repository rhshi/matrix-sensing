import numpy as np 
from sklearn.preprocessing import normalize


class MatrixSensing(object):
	def __init__(self, d, r, eta):
		m = 5 * d * r
		self.matrices = [np.random.randn(d, d) for _ in range(m)]

		U = np.random.randn(d, r)
		U_normalized = normalize(U, norm='l2', axis=0)

		self.X_star = np.matmul(U_normalized, U_normalized.T)

		self.eta = eta
		self.d = d

	def M_t(self, U):
		total = np.zeros((self.d, self.d))
		for A in self.matrices:
			X = np.matmul(U, U.T)
			matrix_product = np.matmul(A.T, X - self.X_star)
			total += np.linalg.norm(matrix_product) * A

		return total / len(self.matrices)

	def step(self, U)
		return np.matmul(np.identity - self.eta * self.M_T(U), U)

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
		U = alpha * np.identity(self.d)
		for _ in range(iters):
			U = self.step(U)

		return U