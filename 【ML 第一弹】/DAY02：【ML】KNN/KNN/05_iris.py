# 1. 依赖包
from sklearn.datasets import load_iris # 鸢尾花数据集
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.neighbors import KNeighborsClassifier # KNN算法
from sklearn.metrics import accuracy_score # 准确率

# 2. 导入鸢尾花数据
data_iris = load_iris()

def show_data_iris(data_iris):

    print(f"数据集信息: {data_iris.keys()}")
    print(f"特征名称: {data_iris.feature_names}")
    print(f"标签名称: {data_iris.target_names}")
    print(f"特征数据: {data_iris.data}")
    print(f"标签数据: {data_iris.target}")

# show_data_iris(data_iris)

# 3. 可视化数据
def visualize_data_iris(data_iris):
    df_iris = pd.DataFrame(data_iris['data'], columns=data_iris.feature_names)
    df_iris['label'] = data_iris.target
    print(f"df_iris: {df_iris.head()}")
    print(f"data_iris.feature_names: {data_iris.feature_names}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten() # 将axes从2x3的二维数组转换为1维数组
    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='label', data=df_iris, ax=axes[0], palette='Set1')
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='label', data=df_iris, ax=axes[1], palette='Set1')
    sns.scatterplot(x='sepal length (cm)', y='petal width (cm)', hue='label', data=df_iris, ax=axes[2], palette='Set1')
    sns.scatterplot(x='sepal width (cm)', y='petal length (cm)', hue='label', data=df_iris, ax=axes[3], palette='Set1')
    sns.scatterplot(x='sepal width (cm)', y='petal width (cm)', hue='label', data=df_iris, ax=axes[4], palette='Set1')
    sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='label', data=df_iris, ax=axes[5], palette='Set1')
    plt.show()

# visualize_data_iris(data_iris)

# 4. 划分数据
def split_data_iris(data_iris):
    X_train, X_test, y_train, y_test = train_test_split(data_iris.data, data_iris.target, test_size=0.2, random_state=42)
    print(f"鸢尾花总数据量：{len(data_iris.data)}")
    print(f"训练集数据量：{len(X_train)}")
    print(f"测试集数据量：{len(X_test)}")

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data_iris(data_iris)

standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train) # 训练集标准化
X_test = standardScaler.transform(X_test) # 测试集标准化

# 5. 训练模型
def train_model_iris(X_train, y_train):
    ## 5.1 实例化模型
    knn_iris = KNeighborsClassifier(n_neighbors=5)
    ## 5.2 训练模型
    knn_iris.fit(X_train, y_train)

    return knn_iris

model_iris = train_model_iris(X_train, y_train)

# 6. 预测模型
def predict_model_iris(model_iris, X_predict):
    y_predict = model_iris.predict(X_predict)
    y_predict_proba = model_iris.predict_proba(X_predict)

    return y_predict, y_predict_proba

X_predict = [[5.1, 3.5, 1.4, 0.2], [4.6, 3.1, 1.5, 0.2]]
X_predict = standardScaler.transform(X_predict) # 预测数据标准化

y_predict, y_predict_proba = predict_model_iris(model_iris, X_predict)
print(f"预测结果：{y_predict}")
print(f"预测概率：{y_predict_proba}")

# 7. 评估模型
def evaluate_model_iris(model_iris, X_test, y_test):
    ## 7.1 直接计算
    acc_1 = model_iris.score(X_test, y_test)
    print(f"直接计算准确率：{acc_1}")
    ## 7.2 计算测试和预测
    y_predict = model_iris.predict(X_test)
    acc_2 = accuracy_score(y_test, y_predict)
    print(f"计算测试和预测准确率：{acc_2}")

evaluate_model_iris(model_iris, X_test, y_test)
