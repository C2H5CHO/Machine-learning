import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ML_basic_function import *

# 1. 准备数据
np.random.seed(42)

features, labels = arrayGenReg(delta=0.01)
print(f"特征: {features[:10]}")
print(f"标签: {labels[:10]}")

# 2. 构造损失函数
w = np.random.randn(3).reshape(-1, 1)
print(f"初始参数: {w}")
y_hat = features.dot(w)
print(f"初始预测结果: {y_hat[:10]}")

mse = MSELoss(features, w, labels)
print(f"初始均方误差: {mse}")
sse = SSELoss(features, w, labels)
print(f"初始误差平方和: {sse}")

# 3. 最小二乘法求解损失函数
## 3.1 inv函数求解
w = np.linalg.inv(features.T.dot(features)).dot(features.T).dot(labels)
print(f"inv求解参数: {w}")
print(f"最小二乘法求解均方误差: {MSELoss(features, w, labels)}")
print(f"最小二乘法求解误差平方和: {SSELoss(features, w, labels)}")
## 3.2 lstsq函数求解
w = np.linalg.lstsq(features, labels, rcond=-1)[0]
print(f"lstsq求解参数: {w}")
