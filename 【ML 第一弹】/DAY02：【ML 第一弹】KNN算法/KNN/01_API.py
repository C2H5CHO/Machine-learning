# 1. 工具包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 2. 数据
# 分类
x = [[0,2,3],[1,3,4],[3,5,6],[4,7,8],[2,3,4]]
y = [0,0,1,1,0]
# 回归
m = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
n = [0.1,0.2,0.3,0.4]

# 3. 实例化
# 分类
model_1 = KNeighborsClassifier(n_neighbors=3)
# 回归
model_2 = KNeighborsRegressor(n_neighbors=3)

# 4. 训练
model_1.fit(x,y)
model_2.fit(m,n)

# 5. 预测
print("分类：", model_1.predict([[4,4,5]]))
print("回归：", model_2.predict([[4,4,5]]))