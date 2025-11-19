# 1. 依赖包
from sklearn.preprocessing import StandardScaler

# 2. 输入数据
X = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3. 特征预处理——标准化
def standardization(X):
    # 3.1 初始化标准化器
    std = StandardScaler()
    # 3.2 拟合并转换数据
    X_std = std.fit_transform(X)

    return X_std

X_std = standardization(X)
print(f"标准化后的数据：{X_std}")
