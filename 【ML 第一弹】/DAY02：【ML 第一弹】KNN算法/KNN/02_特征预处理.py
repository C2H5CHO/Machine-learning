# 1. 工具包
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2. 数据
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3. 实例化
# 归一化
process_1 = MinMaxScaler()
# 标准化
process_2 = StandardScaler()

# 4. 特征处理
res_1 = process_1.fit_transform(x)
res_2 = process_2.fit_transform(x)

print("归一化：", res_1.mean(), res_1.std())
print("标准化：", res_2.mean(), res_2.std())