import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ML_basic_function import *

np.random.seed(42) # 随机种子

# 1. 添加较小的扰动项
features_less, labels_less = arrayGenReg(delta=0.01)
print(f"特征: {features_less[:10]}")
print(f"标签: {labels_less[:10]}")

plt.subplot(121)
plt.scatter(features_less[:, 0], labels_less, c='r')
plt.subplot(122)
plt.scatter(features_less[:, 1], labels_less, c='pink')
plt.show()

# 2. 添加较大的扰动项
features_more, labels_more = arrayGenReg(delta=2)
print(f"特征: {features_more[:10]}")
print(f"标签: {labels_more[:10]}")

plt.subplot(121)
plt.scatter(features_more[:, 0], labels_more, c='b')
plt.subplot(122)
plt.scatter(features_more[:, 1], labels_more, c='g')
plt.show()

# 3. 二阶关系
features_quad, labels_quad = arrayGenReg(deg=2)
print(f"特征: {features_quad[:10]}")
print(f"标签: {labels_quad[:10]}")

plt.subplot(121)
plt.scatter(features_quad[:, 0], labels_quad, c='y')
plt.subplot(122)
plt.scatter(features_quad[:, 1], labels_quad, c='c')
plt.show()
