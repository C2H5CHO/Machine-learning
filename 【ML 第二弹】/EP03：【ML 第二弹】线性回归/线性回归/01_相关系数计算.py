import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. 正变化关系
A = np.array([[1, 2, 3], [4, 5, 10]]).T
print(f"A: {A}")
print(f"A[:, 0]: {A[:, 0]}")
print(f"A的相关系数: {np.corrcoef(A[:, 0], A[:, 1])}")

print('=='*50)
# 2. 负变化关系
B = np.array([[1, 2, 3], [-1, -1.5, -5]]).T
print(f"B: {B}")
print(f"B的相关系数: {np.corrcoef(B[:, 0], B[:, 1])}")
plt.plot(B[:, 0], B[:, 1])
plt.show()

print('=='*50)
# 3. 双变量
C = np.random.randn(20)
y_c = C + 1
print(f"C和y_c的相关系数: {np.corrcoef(C, y_c)}")
plt.plot(C, y_c, 'bo')
plt.show()

print('=='*50)
# 4. 扰动项
D = np.random.randn(20)
y_d = D + 1
print(f"D和y_d的相关系数: {np.corrcoef(D, y_d)}")

ran = np.random.normal(size=D.shape) # 生成随机扰动项
delta = 0.5 # 生成扰动项系数
r = ran*delta # 计算终扰动项
y_d_ = y_d + r # 添加扰动项
print(f"D和y_d_的相关系数: {np.corrcoef(D, y_d_)}")

plt.subplot(121)
plt.plot(D, y_d, 'bo')
plt.title("y_d = D + 1")
plt.subplot(122)
plt.plot(D, y_d_, 'bo')
plt.title("y_d_ = D + 1 + r")
plt.show()

print('=='*50)
# 5. 不同相关系数的双变量
E = np.random.randn(20)
y_e = E + 1
ran = np.random.normal(size=D.shape) # 生成随机扰动项
delta_lst = [0.5, 0.7, 1, 1.5, 2, 5] # 生成扰动项系数

y_lst = [] # 不同delta下y的取值
c_lst = [] # 不同y下相关系数的取值
for i in delta_lst:
    y_n = E + 1 + (ran*i)
    c_lst.append(np.corrcoef(E, y_n))
    y_lst.append(y_n)

plt.subplot(231)
plt.plot(E, y_lst[0], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[0]}")
plt.subplot(232)
plt.plot(E, y_lst[1], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[1]}")
plt.subplot(233)
plt.plot(E, y_lst[2], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[2]}")
plt.subplot(234)
plt.plot(E, y_lst[3], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[3]}")
plt.subplot(235)
plt.plot(E, y_lst[4], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[4]}")
plt.subplot(236)
plt.plot(E, y_lst[5], 'bo')
plt.plot(E, y_e, 'r-')
plt.title(f"delta = {delta_lst[5]}")
plt.show()

print(f"不同y下相关系数的取值: {c_lst}")