import numpy as np
import pandas as pd

# 1. 输入矩阵A、B
A = np.array([[20, 8], [8, 4]])
print(f"A: {A}")
B = np.array([[28, 12]]).T
print(f"B: {B}")

print('=='*50)
# 2. 验证矩阵A是否满秩
## 2.1 矩阵的秩
print(f"np.linalg.matrix_rank(A): {np.linalg.matrix_rank(A)}")
## 2.2 矩阵的行列式
print(f"np.linalg.det(A): {np.linalg.det(A)}")

print('=='*50)
# 3. A的逆矩阵
A_inv = np.linalg.inv(A)
print(f"A_inv: {A_inv}")

print('=='*50)
# 4. 求解方程AX=B
## 4.1 矩阵乘法
X_4_1 = np.matmul(A_inv, B)
print(f"X: {X_4_1}")
## 4.2 向量点积
X_4_2 = A_inv.dot(B)
print(f"X: {X_4_2}")
## 4.3 LU分解
X_4_3 = np.linalg.solve(A, B)
print(f"X: {X_4_3}")
