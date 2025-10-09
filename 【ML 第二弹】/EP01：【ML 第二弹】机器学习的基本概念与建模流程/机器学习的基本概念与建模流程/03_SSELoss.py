import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 创建w和b的取值范围，从-1到3，步长为0.05
x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)
# 创建网格点，用于绘制3D图形
w, b = np.meshgrid(x, y)
# 计算SSE损失函数的值
SSELoss = (2-w-b)**2 + (4-3*w-b)**2

# 创建3D图形对象
ax = plt.axes(projection='3d')
# 绘制3D曲面图，使用彩虹色图
ax.plot_surface(w, b, SSELoss, cmap='rainbow', edgecolor='none')
# 在z=0平面上绘制等高线图
ax.contour(w, b, SSELoss, zdir='z', offset=0, cmap='rainbow')
# 设置x轴标签
ax.set_xlabel('w')
# 设置y轴标签
ax.set_ylabel('b')
# 显示图形
plt.show()