import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. y = 2*x_1 - x_2 + 1
np.random.seed(24) # 随机种子

w_true = np.array([2, -1]).reshape(-1, 1) # 参数
b_true = np.array(1) # 偏置
delta = 0.01 # 扰动项

num_inputs = 2 # 输入维度
num_examples = 1000 # 样本数量
features = np.random.randn(num_examples, num_inputs) # 生成特征
labels_true = features.dot(w_true) + b_true # 生成标签
labels_less = labels_true + np.random.normal(size=labels_true.shape) * delta # 添加较小的扰动项
labels_more = labels_true + np.random.normal(size=labels_true.shape) * 2 # 添加较大的扰动项

print(f"特征: {features[:10]}")
print(f"标签: {labels_true[:10]}")

plt.subplot(221)
plt.scatter(features[:, 0], labels_less, label='labels_less')
plt.subplot(222)
plt.plot(features[:, 1], labels_less, 'go')
plt.subplot(223)
plt.scatter(features[:, 0], labels_more, label='labels_more')
plt.subplot(224)
plt.plot(features[:, 1], labels_more, 'yo')
plt.show()

print('=='*50)
# 2. y = 2*x^2 + 1
np.random.seed(42) # 随机种子

w_true = np.array(2) # 参数
b_true = np.array(1) # 偏置

num_inputs = 1 # 输入维度
num_examples = 1000 # 样本数量
features = np.random.randn(num_examples, num_inputs) # 生成特征
labels_true = features ** 2 * w_true + b_true # 生成标签
labels_ = labels_true + np.random.normal(size=labels_true.shape) * 2 # 添加扰动项

plt.scatter(features, labels_)
plt.show()

