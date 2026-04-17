from scipy import sparse
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6])
row = np.array([0, 0, 1, 1, 2, 3])
col = np.array([1, 2, 0, 1, 2, 0])

sparse_coo = sparse.coo_matrix((data, (row, col)))

print(type(sparse_coo))
print(sparse_coo)
dense_mat = sparse_coo.toarray()
print(type(dense_mat))
print(dense_mat)


data = np.array([1, 2, 3, 4, 5, 6])
row = np.array([0, 2, 4, 5, 6])
col = np.array([1, 2, 0, 1, 2, 0])

sparse_csr = sparse.csr_matrix((data, col, row))
print(type(sparse_csr))
print(sparse_csr)
dense_mat = sparse_csr.toarray()
print(type(dense_mat))
print(dense_mat)

