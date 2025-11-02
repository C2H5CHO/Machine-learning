from sklearn.neighbors import KNeighborsRegressor

def knn_regress(X_train, y_train, X_test, n_neighbors=5):
    """
    KNN regressor
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征
    :param n_neighbors: KNN参数
    :return: y_pred
    """
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

if __name__ == '__main__':
    X_train = [[0, 0, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]
    y_train = [0.1, 0.2, 0.3, 0.4]
    X_test = [[3, 11, 10]]
    y_pred = knn_regress(X_train, y_train, X_test, n_neighbors=3)
    print(f"测试集标签：{y_pred}")
