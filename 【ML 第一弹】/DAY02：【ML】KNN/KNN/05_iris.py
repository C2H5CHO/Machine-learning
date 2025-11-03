# 1. 依赖包
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
