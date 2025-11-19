# 1. 依赖包
from sklearn.preprocessing import MinMaxScaler

# 2. 输入数据
X = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3. 特征预处理——归一化
def Normalization(X_train):
    ## 3.1 实例化模型
    scaler = MinMaxScaler(feature_range=(0, 1))
    ## 3.2 归一化
    X_train = scaler.fit_transform(X_train)

    return X_train

X_train = Normalization(X)
print(f"X_train: {X_train}")
