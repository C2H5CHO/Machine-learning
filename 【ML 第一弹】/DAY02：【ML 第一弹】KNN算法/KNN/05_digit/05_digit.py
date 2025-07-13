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