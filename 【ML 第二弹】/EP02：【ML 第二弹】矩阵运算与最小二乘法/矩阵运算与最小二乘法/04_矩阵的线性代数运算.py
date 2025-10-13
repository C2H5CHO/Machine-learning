import numpy as np
import pandas as pd

# 1. 矩阵的迹
A_1 = np.array([[1, 2], [4, 5]])
print(f"A_1: {A_1}")
print(f"np.trace(A_1): {np.trace(A_1)}")

A_1_1 = np.arange(1, 7).reshape(2, 3)
print(f"A_1_1: {A_1_1}")
print(f"np.trace(A_1_1): {np.trace(A_1_1)}")

print('=='*50)
# 2. 矩阵的秩
A_2 = np.array([[1, 3, 4], [2, 1, 3], [1, 1, 2]])
print(f"A_2: {A_2}")
print(f"np.linalg.matrix_rank(A_2): {np.linalg.matrix_rank(A_2)}")

A_2_1 = np.array([[1, 3, 4], [2, 1, 3], [1, 1, 10]])
print(f"A_2_1: {A_2_1}")
print(f"np.linalg.matrix_rank(A_2_1): {np.linalg.matrix_rank(A_2_1)}")

print('=='*50)
# 3. 矩阵的行列式
A_3 = np.array([[1, 2], [4, 5]])
print(f"A_3: {A_3}")
print(f"np.linalg.det(A_3): {np.linalg.det(A_3)}")

A_3_1 = np.arange(1, 7).reshape(2, 3)
print(f"A_3_1: {A_3_1}")
# print(f"np.linalg.det(A_3_1): {np.linalg.det(A_3_1)}")
# numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square

A_3_2 = np.array([[1, 3, 4], [2, 1, 3], [1, 1, 2]])
print(f"A_3_2: {A_3_2}")
print(f"np.linalg.det(A_3_2): {np.linalg.det(A_3_2)}")

print('=='*50)
# 4. 矩阵的逆
A_4 = np.array([[1, 1], [3, 1]])
print(f"A_4: {A_4}")
print(f"np.linalg.inv(A_4): {np.linalg.inv(A_4)}")
