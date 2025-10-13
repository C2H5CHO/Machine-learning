import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 10]]).T
print(f"A: {A}")
print(f"A[:, 0]: {A[:, 0]}")
print(f"A的相关系数: {np.corrcoef(A[:, 0], A[:, 1])}")

B = np.array([[1, 2, 3], [-1, -1.5, -5]]).T
print(f"B: {B}")
print(f"B的相关系数: {np.corrcoef(B[:, 0], B[:, 1])}")
plt.plot(B[:, 0], B[:, 1])
plt.show()

