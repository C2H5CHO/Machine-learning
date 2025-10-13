import numpy as np
import pandas as pd

# 1. 四则乘法
A_1 = np.arange(4)
print(f"A_1: {A_1}")
print(f"A_1 * A_1: {A_1 * A_1}")

A_1_ = A_1.reshape(2, 2)
print(f"A_1_: {A_1_}")
print(f"A_1_ * A_1_: {A_1_ * A_1_}")

print('=='*50)
# 2. 向量点积
A_2 = np.arange(4)
A_dot = np.dot(A_2, A_2)
print(f"A_dot: {A_dot}")
print(f"A_2.dot(A_2): {A_2.dot(A_2)}")
print(f"np.vdot(A_2, A_2): {np.vdot(A_2, A_2)}")
print(f"np.inner(A_2, A_2): {np.inner(A_2, A_2)}")

print('=='*50)
# 3. 矩阵点积
A_3 = np.arange(4).reshape(2, 2)
print(f"A_3: {A_3}")
print(f"np.vdot(A_3): {np.vdot(A_3, A_3)}")
print(f"(A_3 * A_3).sum(): {(A_3 * A_3).sum()}")

print('=='*50)
# 4. 矩阵乘法
A_4 = np.arange(1, 7).reshape(2, 3)
print(f"A_4: {A_4}")
A_4_1 = np.arange(1, 10).reshape(3, 3)
print(f"A_4_1: {A_4_1}")
print(f"np.matmul(A_4, A_4_1): {np.matmul(A_4, A_4_1)}")
