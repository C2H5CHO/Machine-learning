# 一、算法简介
## 1.1 算法思想
如果一个样本在特征空间中的 k 个最相似的样本中的大多数属于某一个类别，则该样本也属于这个类别。
## 1.2 样本相似性
样本都是属于一个任务数据集的，样本距离越近则越相似。
1. 二维平面上点的欧氏距离
二维平面上点 $a(x_1, y_1)$ 与 $b(x_2, y_2)$ 间的欧氏距离：
$$
d_{12} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$
2. 三维空间点的欧氏距离
三维空间点 $a(x_1, y_1, z_1)$ 与 $b(x_2, y_2, z_2)$ 间的欧氏距离：
$$
d_{12} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}
$$
3. n维空间点（向量）的欧氏距离
n维空间点 $a(x_{11}, x_{12}, \dots, x_{1n})$ 与 $b(x_{21}, x_{22}, \dots, x_{2n})$ 间的欧氏距离（两个n维向量）：
$$
d_{12} = \sqrt{\sum_{k=1}^{n} (x_{1k} - x_{2k})^2}
$$
## 1.3 K值的选择
### 1.3.1 大小选择
1. K值过小：
- 即用**较小领域**中的训练实例进行预测
- 易受到**异常点**的影响
- 意味着整体模型变得复杂，容易发生**过拟合**
2. K值过大：
- 即用较大领域中的训练实例进行预测
- 易受到**样本均衡**的问题
- 意味着整体模型变得简单，容易发生**欠拟合**
### 1.3.2 方法选择
- 交叉验证
- 网格搜索
## 1.4 应用方式
### 1.4.1 分类问题
1. 计算未知样本到每一个训练样本的距离
2. 将训练样本根据距离大小升序排列
3. 取出距离最近的 K 个训练样本
4. 进行**多数表决**，统计 K 个样本中哪个类别的样本个数最多
5. 将未知的样本归属到**出现次数最多的类别**
### 1.4.2 回归问题
1. 计算未知样本到每一个训练样本的距离
2. 将训练样本根据距离大小升序排列
3. 取出距离最近的 K 个训练样本
4. 把这个 K 个样本的目标值计算其平均值
5. 作为将未知的样本预测的值
# 二、API简介
## 2.1 分类问题

```python
class sklearn.neighbors.KNeighborsClassifier( 
    n_neighbors=5, 
    weights='uniform', 
    algorithm='auto', 
    leaf_size=30, 
    p=2, 
    metric='minkowski', 
    metric_params=None, 
    n_jobs=None 
)
```
### 2.1.1 参数说明
- **n_neighbors (int, default=5)** ：表示K值 ，即预测样本时考虑的最近邻的数量。
- **weights ({'uniform', 'distance'} or callable, default='uniform')** ：权重函数，用于确定在预测时，近邻样本对预测结果的影响程度。
    - `'uniform'`：所有邻居的权重相同，即每个近邻样本在预测中具有同等的影响力。
    - `'distance'`：权重与距离成反比，意味着距离预测样本越近的邻居，对预测结果的影响越大。
    - 自定义一个可调用的函数，根据距离来计算每个邻居的权重。
- **algorithm ({'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto')** ：计算最近邻的算法。
    - `'brute'`：暴力搜索算法，它会计算所有样本之间的距离，然后找出K个最近邻。
    - `'kd_tree'`：KD树算法，是一种对k维空间中的实例点进行存储以便快速检索的树形数据结构，适用于低维数据，一般维数小于20时效果较好。
    - `'ball_tree'`：球树算法，通过超球体来划分样本空间，每个节点对应一个超球体，相比KD树在高维数据上表现更优。
    - `'auto'`：自动选择最合适的算法，算法会根据训练数据的规模、特征维度等因素来选择。
- **leaf_size (int, default=30)** ：仅在使用`'kd_tree'`或`'ball_tree'`算法时生效，表示KD树或球树的叶子节点大小。
- **p (int, default=2)** ：距离度量的参数，仅当`metric='minkowski'`时有效。
    - `p=1`：表示曼哈顿距离（L1范数），计算公式为$d(x,y)=\sum_{i=1}^{n}|x_i - y_i|$ 。
    - `p=2`：表示欧氏距离（L2范数），计算公式为$d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$ 。
- **metric (str or callable, default='minkowski')** ：距离度量类型。
    - `'euclidean'`（欧氏距离）
    - `'manhattan'`（曼哈顿距离）
    - `'chebyshev'`（切比雪夫距离）
    - `'minkowski'`（闵可夫斯基距离）
    - 自定义一个可调用的函数，计算样本之间的距离。
 - **metric_params (dict, default=None)** ：距离度量的额外参数，当使用自定义距离度量函数或者某些需要额外参数的距离度量时使用。
 - **n_jobs (int, default=None)** ：并行计算数。
 	- -1，表示使用所有可用的处理器进行并行计算 ，以加快计算速度。
 	- 具体的整数，表示使用指定数量的处理器进行并行计算。 
### 2.1.2 常用方法
- **fit(X, y)** ：用于拟合模型，将训练数据`X`（特征矩阵，形状为`(n_samples, n_features)`）和对应的标签`y`（形状为`(n_samples,)`）输入该方法，模型会存储训练数据。
- **predict(X)** ：预测输入数据`X`（特征矩阵）的类别标签，返回一个数组，数组中的每个元素是对应样本的预测类别。
- **predict_proba(X)** ：返回输入数据`X`属于各类别的概率，返回一个形状为`(n_samples, n_classes)`的数组，`n_samples`是样本数量，`n_classes`是类别数量，数组中每个元素`[i][j]`表示第`i`个样本属于第`j`个类别的概率。
- **kneighbors((X, n_neighbors, return_distance))** ：查找点的K近邻。
    - `X`：需要查找近邻的样本点（特征矩阵）。
    - `n_neighbors`：指定查找的近邻数量，如果不指定，则使用构造函数中设置的`n_neighbors`值。
    - `return_distance`：布尔值，默认为`True`，表示是否返回距离信息。如果为`True`，会返回两个数组，第一个数组是查询点与近邻点之间的距离，第二个数组是近邻点在训练数据中的索引；如果为`False`，只返回近邻点在训练数据中的索引。 
- **score(X, y)** ：返回给定测试数据`X`和标签`y`的平均准确率，即预测正确的样本数占总样本数的比例，用于评估模型在测试集上的性能。
### 2.1.3 代码实操

```python
# 1. 工具包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 2. 数据
# 分类
x = [[0,2,3],[1,3,4],[3,5,6],[4,7,8],[2,3,4]]
y = [0,0,1,1,0]

# 3. 实例化
# 分类
model_1 = KNeighborsClassifier(n_neighbors=3)

# 4. 训练
model_1.fit(x,y)

# 5. 预测
print("分类：", model_1.predict([[4,4,5]]))
```
## 2.2 回归问题

```python
class sklearn.neighbors.KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=None
)
```
### 2.2.1 代码实操

```python
# 1. 工具包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 2. 数据
# 回归
m = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
n = [0.1,0.2,0.3,0.4]

# 3. 实例化
# 回归
model_2 = KNeighborsRegressor(n_neighbors=3)

# 4. 训练
model_2.fit(m,n)

# 5. 预测
print("回归：", model_2.predict([[4,4,5]]))
```
# 三、距离度量方法
## 3.1 欧氏距离
### 3.1.1 定义
- Euclidean Distance 欧氏距离
- 两个点在空间中的距离
### 3.1.2 数学公式
1. 二维平面上点的欧氏距离
二维平面上点 $a(x_1, y_1)$ 与 $b(x_2, y_2)$ 间的欧氏距离：
$$
d_{12} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$
2. 三维空间点的欧氏距离
三维空间点 $a(x_1, y_1, z_1)$ 与 $b(x_2, y_2, z_2)$ 间的欧氏距离：
$$
d_{12} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + (z_1 - z_2)^2}
$$
3. n维空间点（向量）的欧氏距离
n维空间点 $a(x_{11}, x_{12}, \dots, x_{1n})$ 与 $b(x_{21}, x_{22}, \dots, x_{2n})$ 间的欧氏距离（两个n维向量）：
$$
d_{12} = \sqrt{\sum_{k=1}^{n} (x_{1k} - x_{2k})^2}
$$
## 3.2 曼哈顿距离
### 3.2.1 定义
- Manhattan Distance 曼哈顿距离
- City Block Distance 城市街区距离
- **横平竖直**
### 3.2.2 数学公式
1. 二维平面两点 $a(x_1,y_1)$ 与 $b(x_2,y_2)$ 间的曼哈顿距离：
$$
d_{12} =|x_1 - x_2| + |y_1 - y_2|
$$
2. n维空间点 $a(x_{11},x_{12},\dots,x_{1n})$ 与 $b(x_{21},x_{22},\dots,x_{2n})$ 的曼哈顿距离：
$$
d_{12} = \sum_{k=1}^{n} |x_{1k} - x_{2k}|
$$
## 3.3 切比雪夫距离
### 3.3.1 定义
- Chebyshev Distance 切比雪夫距离
- 国际象棋中，国王可以直行、横行、斜行，所以国王走一步可以移动到相邻8个方格中的任意一个。国王从格子 $(x_1,y_1)$ 走到格子 $(x_2,y_2)$ 最少需要的步数
### 3.3.2 数学公式
1. 二维平面两点 $\text{a}(x_1,y_1)$ 与 $\text{b}(x_2,y_2)$ 间的切比雪夫距离：
$$
d_{12} = \max\left(|x_1 - x_2|, |y_1 - y_2|\right)
$$
2. n维空间点 $\text{a}(x_{11},x_{12},\dots,x_{1n})$ 与 $\text{b}(x_{21},x_{22},\dots,x_{2n})$ 的切比雪夫距离：
$$
d_{12} = \max\left(|x_{1i} - x_{2i}|\right) \quad (i = 1,2,\dots,n)
$$
## 3.4 闵氏距离
### 3.4.1 定义
- Minkowski Distance 闵可夫斯基距离
- 根据 p 的不同，闵氏距离可表示某一种类的距离
### 3.4.2 数学公式
两个n维变量 $\boldsymbol{a}(x_{11}, x_{12}, \dots, x_{1n})$ 与 $\boldsymbol{b}(x_{21}, x_{22}, \dots, x_{2n})$ 间的闵可夫斯基距离定义为：
$$
d_{12} = \sqrt[p]{\sum_{k=1}^{n} \left| x_{1k} - x_{2k} \right|^p}
$$
- 变参数 p：
	- 当 $p = 1$ 时，退化为**曼哈顿距离**（Manhattan Distance）
	- 当 $p = 2$ 时，退化为**欧氏距离**（Euclidean Distance）
	- 当 $p \to \infty$ 时，退化为**切比雪夫距离**（Chebyshev Distance）
# 四、特征预处理
## 4.1 预处理的原因
特征的**单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级**，容易影响（支配）目标结果，使得一些模型（算法）无法学习到其它的特征。
## 4.2 预处理的方法
### 4.2.1 归一化
1. 定义：通过对原始数据进行变换把数据映射到 [mi, mx]（默认为 [0, 1]）之间。
2. 数学公式：
	- $X' = \frac{x - \min}{\max - \min}$
	- $X'' = X' * (mx - mi) + mi$
3. API：`sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))`
	- `feature_range=(min, max)`：指定归一化后数据的取值范围，`min` 和 `max` 为自定义区间上下限。
4. 特点：
	- 受到最大值和最小值的影响
	- 容易受到异常数据的影响，鲁棒性较差
	- 适合传统精确小数据的场景
5. 代码实操：

```python
# 1. 工具包
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2. 数据
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3. 实例化
# 归一化
process_1 = MinMaxScaler()

# 4. 特征处理
res_1 = process_1.fit_transform(x)

print("归一化：", res_1.mean(), res_1.std())
```
### 4.2.2 标准化
1. 定义：通过对原始数据进行标准化，转换为均值为0、标准差为1的标准正态分布的数据。
2. 数学公式：
	- $X' = \frac{x - \text{mean}}{\sigma}$
	- $mean$：特征平均值
	- $\sigma$：特征标准差
3. API：`sklearn.preprocessing.StandardScaler()`
	- `with_mean=True`：是否中心化（均值为 0）
	- `with_std=True`：是否缩放（标准差为 1）
4. 特点：
	- 如果出现异常点，由于具有一定数据量，**少量**的异常点对于平均值的影响并不大
	- 适合现代嘈杂大数据场景
5. 代码实操：

```python
# 1. 工具包
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2. 数据
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3. 实例化
# 标准化
process_2 = StandardScaler()

# 4. 特征处理
res_1 = process_1.fit_transform(x)

print("标准化：", res_2.mean(), res_2.std())
```
## 4.3 代码实操
### 4.3.1 利用KNN算法进行鸢尾花分类

```python
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
```
### 4.3.2 利用KNN算法实现手写数字识别

```python
# 1. 工具包
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

# 2. 加载数据集
digit = pd.read_csv('手写数字识别.csv')
# 显示指定的信息
x = digit.iloc[:, 1:]
y = digit.iloc[:, 0]
print("基本信息：", x.shape)
print("类别比例：", Counter(y))
print("第一个数字标签：", y[0])
# 显示指定的图片
digit_ = x.iloc[0].values
digit_ = digit_.reshape(28, 28)
plt.imshow(digit_, cmap='gray')
plt.show()

# 3. 归一化处理
x = digit.iloc[:, 1:] / 255
y = digit.iloc[:, 0]

# 4. 划分数据集
X_train, X_test,Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# 5. 模型训练
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)

# 6. 模型评估
acc = model.score(X_test, Y_test)
print(f"准确率：{acc:.2f}")

# 7. 模型保存
joblib.dump(model, 'KNN.pth')

# 8. 模型预测
# 加载测试图片
img = plt.imread('demo.png')
plt.imshow(img)
plt.show()
# 加载保存后的模型
KNN = joblib.load('KNN.pth')
# 预测测试图片
y_pred = KNN.predict(img.reshape(1, -1))
print(f"该图中的数字是：{y_pred}")
```

# 五、超参数选择方法
## 5.1 交叉验证
### 5.1.1 定义
&emsp;&emsp;一种数据集的分割方法，将训练集划分为 n 份，拿一份做验证集（测试集）、其他 n-1 份做训练集。
### 5.1.2 原理
（假设数据集划分为：`cv=4`）
1. 第一次：把第一份数据做验证集，其他数据做训练。
2. 第二次：把第二份数据做验证集，其他数据做训练。
3. ... 以此类推，总共训练4次，评估4次。
4. 使用**训练集 + 验证集**多次评估模型，取平均值做交叉验证为模型得分。
5. 若 `k=5` 模型得分最好，再使用全部数据集（训练集 + 验证集）对 `k=5` 模型再训练一边，再使用测试集对 `k=5` 模型做评估。
### 5.1.3 目的
- 为了得到更加准确可信的模型评分。
## 5.2 网格搜索
### 5.2.1 定义
&emsp;&emsp;模型调参的有力工具，寻找最优超参数的工具！只需要将若干参数传递给网格搜索对象，它自动帮我们完成不同超参数的组合、模型训练、模型评估，最终返回一组最优的超参数。
### 5.2.2 目的
- 模型有很多超参数，其能力也存在很大的差异，需要手动产生很多超参数组合，来训练模型。
- 每组超参数都采用交叉验证评估，最后选出最优参数组合建立模型。
## 5.3 API简介

```python
sklearn.model_selection.GridSearchCV(
    estimator,
    param_grid=None,
    cv=None,
    scoring='accuracy',
    n_jobs=-1
)
```
1. 参数：
- `estimator`：待调优的模型（如 SVC、RandomForestClassifier）。
- `param_grid`：参数字典，格式为{参数名: [取值列表]}。
- `cv`：交叉验证折数（整数或 CV 迭代器）。
- `scoring`：评估指标（如`accuracy`、`f1`、`roc_auc`）。
- `n_jobs`：并行计算的 CPU 核心数，-1表示全部使用。
2. 方法：
- `fit(X, y)`：执行网格搜索和交叉验证。
- `predict(X)`：使用最佳模型预测。
- `best_params_`：最佳参数组合。
- `best_score_`：最佳模型在验证集上的平均分数。
- `cv_results_`：所有参数组合的详细结果。
## 5.4 代码实操——利用KNN算法进行鸢尾花分类(2)

```python
# 1. 工具包
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. 加载数据集
iris = load_iris()

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 4. 特征预处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. 模型实例化 + 交叉验证 + 网格搜索
model = KNeighborsClassifier(n_neighbors=1)
param_grid = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9]}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=4,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
# 最优参数和最佳模型性能
print(f"最佳参数: {grid_search.best_params_}")
print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
# 6. 使用最优参数的模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
# 模型评估
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
# 分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# 对新样本进行预测
new_sample = [[5.1, 3.5, 1.4, 0.2]]
new_sample_scaled = scaler.transform(new_sample)
predicted_class = best_model.predict(new_sample_scaled)
predicted_probabilities = best_model.predict_proba(new_sample_scaled)
print(f"预测样本类别: {iris.target_names[predicted_class][0]}")
print(f"各类别概率: {[f'{p:.4f}' for p in predicted_probabilities[0]]}")
```

-----
==微语录：若不趁着风时扬帆，船是不会前进的。——东野圭吾==
