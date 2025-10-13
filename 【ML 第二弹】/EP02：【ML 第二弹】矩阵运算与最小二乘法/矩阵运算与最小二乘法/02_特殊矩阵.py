import numpy as np
import pandas as pd

# 1. 矩阵转置
A_T = np.arange(1, 7).reshape(2, 3)
print(f"A_T: {A_T}")
print(f"A_T.T: {A_T.T}")

print('=='*50)
# 2. 单位矩阵
A_eye = np.eye(3)
print(f"A_eye: {A_eye}")

print('=='*50)
# 3. 对角矩阵
A_3 = np.arange(5)
print(f"A_3: {A_3}")

print('--'*50)
## 3.1 正对角线
A_diag = np.diag(A_3)
print(f"A_diag: {A_diag}")

print('--'*50)
## 3.2 对角线上移一位
A_diag2 = np.diag(A_3, k=1)
print(f"A_diag2: {A_diag2}")

print('--'*50)
## 3.3 对角线下移一位
A_diag3 = np.diag(A_3, k=-1)
print(f"A_diag3: {A_diag3}")

print('=='*50)
# 4. 上/下三角矩阵
A_4 = np.arange(9).reshape(3, 3)
print(f"A_4: {A_4}")

print('--'*50)
## 4.1 上三角矩阵
A_triu = np.triu(A_4)
print(f"A_triu: {A_triu}")
print(f"右上偏移一位: {np.triu(A_4, k=1)}")
print(f"左下偏移一位: {np.triu(A_4, k=-1)}")

print('--'*50)
## 4.2 下三角矩阵
A_tril = np.tril(A_4)
print(f"A_tril: {A_tril}")