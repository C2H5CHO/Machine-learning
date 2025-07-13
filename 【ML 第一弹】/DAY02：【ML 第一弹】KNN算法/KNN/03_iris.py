# 1. 工具包
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 2. 获取数据集
from sklearn.datasets import load_iris

# 3. 加载数据集
iris = load_iris()
# 查看信息
print("信息：", iris.data[:5])
# 查看目标值
print("目标值：", iris.target)
# 查看目标值名字
print("目标值名字：", iris.target_names)
# 查看特征名
print("特征名：", iris.feature_names)
# 查看描述
print("描述：", iris.DESCR)
# 查看文件路径
print("文件路径：", iris.filename)

# 4. 可视化数据集
# 转换数据格式为dataframe
iris_df = pd.DataFrame(iris['data'], columns=iris.feature_names)
iris_df['label'] = iris.target
print("dataframe：", iris_df.head())
# 可视化数据
sns.lmplot(x='sepal length (cm)', y='sepal width (cm)', data=iris_df, hue='label', fit_reg=False)
plt.show()

# 5. 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=22)

# 6. 模型训练和预测
# 数据标准化
process = StandardScaler()
X_train = process.fit_transform(X_train)
X_test = process.transform(X_test)
# 模型训练
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)
# 模型评估
model_score = model.score(X_test, Y_test)
print("准确率：", model_score)
# 模型预测
x = [[5.1, 3.5, 1.4, 0.2]]
x = process.transform(x)
y_predict =model.predict(X_test)
print("预测结果：", model.predict(x))
print("预测概率值：", model.predict_proba(x))