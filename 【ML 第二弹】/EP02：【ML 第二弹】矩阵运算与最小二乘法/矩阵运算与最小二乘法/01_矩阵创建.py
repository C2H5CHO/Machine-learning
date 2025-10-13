import numpy as np
import pandas as pd

# 1. 利用数组创建矩阵
A_array = np.array([[1, 2], [1, 1]])
print(f"A_array: {A_array}")
print(f"A_array.type: {type(A_array)}")

print('=='*50)
# 2. 利用mat创建矩阵
A_mat = np.mat(A_array)
print(f"A_mat: {A_mat}")
print(f"A_mat.type: {type(A_mat)}")

print('=='*50)
# 3. 矩阵乘法
## 3.1 数组乘法
print(f"A_array @ A_array: {A_array @ A_array}")
print(f"A_array.dot(A_array): {A_array.dot(A_array)}")
print(f"A_array * A_array: {A_array * A_array}")

print('--'*50)
## 3.2 矩阵乘法
print(f"A_mat * A_mat: {A_mat * A_mat}")
