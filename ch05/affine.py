import numpy as np

X = np.array([1, 2])
W = np.array([[1,1,1], [1,1,1]])


print(str(X.shape))
print(str(W.shape))
print(np.dot(X, W))
