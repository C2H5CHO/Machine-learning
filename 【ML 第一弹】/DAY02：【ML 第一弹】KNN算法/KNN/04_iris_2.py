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