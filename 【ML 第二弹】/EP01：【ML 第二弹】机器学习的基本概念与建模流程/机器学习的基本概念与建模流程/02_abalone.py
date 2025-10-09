import numpy as np
import pandas as pd

# 1. 读取数据
abalone_df = pd.read_csv('./data/abalone.txt', sep='\t', header=None)
print(abalone_df)
print(abalone_df.columns)

print('--'*50)
# 2. 修改列名
abalone_df.columns = ['Gender', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
print(abalone_df)

# 3. 保存数据
abalone_df.to_csv("./data/abalone.csv", index=False)
