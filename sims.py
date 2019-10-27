import numpy as np 
from sklearn.preprocessing import normalize


U = np.random.randn(100, 5)
U_normalized = normalize(U, norm='l2', axis=0)

print(U_normalized)

A = np.matmul(U_normalized, U_normalized.T)
print(A)
print(np.linalg.norm(A, ord=2))
print(np.linalg.matrix_rank(A))