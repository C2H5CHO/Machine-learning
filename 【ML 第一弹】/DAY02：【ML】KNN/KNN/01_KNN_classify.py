from sklearn.neighbors import KNeighborsClassifier

def knn_classify(X_train, Y_train, X_test, k):
    """
    KNN分类器
    :param X_train: 训练集特征
    :param Y_train: 训练集标签
    :param X_test: 测试集特征
    :param k: K值
    :return: 预测结果
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    return Y_pred

if __name__ == '__main__':
    # 测试代码
    X_train = [[0], [1], [2], [3]]
    Y_train = [0, 0, 1, 1]
    X_test = [[4]]
    Y_pred = knn_classify(X_train, Y_train, X_test, 3)
    print(f"预测结果为：{Y_pred}")
