import numpy as np


# 假设X是一个形状为 (m, n) 的矩阵，Y是一个形状为 (m, 1) 的矩阵
X = np.array([[１，８．６３，１４．５，３．６，５，７．３], [3, 4], [5, 6]])
Y = np.array([[7], [8], [9]])

# 计算 (X.T * X) 的逆矩阵乘以 X.T 乘以Y
result = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
print(result)
