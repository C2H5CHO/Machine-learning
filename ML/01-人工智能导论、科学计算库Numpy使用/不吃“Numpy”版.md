# 一、概述
## 1.1 目的
Numpy 补充了 Python 语言所欠缺的数值计算能力
## 1.2 功能
Numpy 提供了对大型数组和矩阵进行计算的功能，是其他数据分析和机器学习的底层库
## 1.3 特点
Numpy 完全基于标准的 C 语言实现的，运行效率得到了充分优化，程序执行速度快，是机器学习的库基础
## 1.4 发展历史
- 1995年，建立了一个数值计算库 Numeric
- 2001年，在 Scipy 库中添加了名为 Numarray 的多维数组计算对象
- 2005年，Numeric + Numarray >> Numpy
- 2006年，Numpy 从 Scipy 中独立成为一个软件包，主要用于科学计算，如信号处理、数值微积分等
## 1.5 核心：多维数组
- 向量：即一维数组
- 矩阵：即二维数组
- 张量：即三维数组

1. ==代码简洁==：减少 Python 代码中的循环
2. ==底层实现==：厚内核（C）+ 薄接口（Python），保证性能
# 二、基础
## 2.1 `ndarray`数组
### 2.1.1 内存对象
#### （1）元数据（metadata）
对目标数组的描述信息，如`dim count`，`dimensions`，`dtype`，`data`等
#### （2）实际数据
完整的数组数据

> Tips：
> 1. 元数据与实际数据分开存放的原因：
> 	
> 		- 提高了内存空间的使用效率
> 		- 减少了对实际数据的访问频率，提高性能
### 2.1.2 特点
1. Numpy 数组是**同质数组**，即所有的数据类型必须相同
2. Numpy 数组的下标从0开始，最后一个元素的下标为数组长度-1
### 2.1.3 创建
1. `np.array()`

```python
arr1 = np.array([1, 2, 3, 4,5])
print(arr1)
print(type(arr1))
"""
[1 2 3 4 5]
<class 'numpy.ndarray'>
"""
```
2. `np.arange(起始值, 终止值, 步长)`

```python
arr2 = np.arange(0, 10, 2) # 0-9，步长为2
print(arr2)
print(type(arr2))
"""
[0 2 4 6 8]
<class 'numpy.ndarray'>
"""
```
3. `np.zeros(数组元素个数, dtype='类型')`，`np.ones(数组元素个数, dtype='类型')`

```python
arr3 = np.zeros(10)
print(arr3)

arr4 = np.ones(10)
print(arr4)
"""
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
"""
```
### 2.1.4 基本操作
- 数组维度：`np.ndarray.shape`
- 元素类型：`np.ndarray.dtype`
- 元素个数：`np.ndarray.size`

```python
arr1 = np.array([
    [
        [1,2,3],
        [4,5,6]
    ],
    [
        [7,8,9],
        [10,11,12]
    ]
])

print("数组形状：", arr1.shape) # 输出数组形状
print("数组维度", arr1.ndim) # 输出数组维度
print('数组元素个数', arr1.size) # 输出数组元素个数
print('数组元素类型', arr1.dtype) # 输出数组元素类型
"""
数组形状： (2, 2, 3)
数组维度 3
数组元素个数 12
数组元素类型 int64
"""
```
- 索引访问

```python
print("第一页：", arr1[0])
print("第一页 第一行：", arr1[0][0])
print("第一页 第一行 第一列：", arr1[0][0][0])
print(arr1[0, 0, 1])
"""
第一页： [[1 2 3]
 [4 5 6]]
第一页 第一行： [1 2 3]
第一页 第一行 第一列： 1
2
"""
```
- for 循环遍历

```python
for i in range(arr1.shape[0]): # 遍历每一页
    for j in range(arr1.shape[1]): # 遍历每一行
        for k in range(arr1.shape[2]): # 遍历每一列
            print(arr1[i][j][k], end=' ')
        print()
"""
1 2 3 
4 5 6 
7 8 9 
10 11 12 
"""
```
### 2.1.5 属性操作
#### （1）基本数据类型
| **类型分类**       | NumPy类型名       | 类型表示符（Python风格）          | 字符码（简洁表示） | 存储大小/说明                                                                 |
| ------------------ | ------------------ | ----------------------------------- | ------------------ | ----------------------------------------------------------------------------- |
| 布尔型             | `np.bool_`         | `bool_`                            | `?`                | 1字节，值为`True`或`False`                                                   |
| 有符号整数型       | `np.int8/16/32/64` | `int8`（-128~127）、`int16`、`int32`、`int64` | `i1/i2/i4/i8`      | 数字后的数字表示字节数，如`i4`对应4字节（32位），范围随字节增大而扩大         |
| 无符号整数型       | `np.uint8/16/32/64`| `uint8`（0~255）、`uint16`、`uint32`、`uint64` | `u1/u2/u4/u8`      | 无负数，范围为`0`到`2^n-1`（n为位数，如`u1`对应8位，范围0~255）             |
| 浮点型             | `np.float16/32/64` | `float16`（半精度）、`float32`（单精度）、`float64`（双精度） | `f2/f4/f8`         | 字节数对应精度，如`f8`为8字节（64位，Python默认浮点类型）                    |
| 复数型             | `np.complex64/128` | `complex64`（实虚部分各32位）、`complex128`（各64位） | `c8/c16`           | 总字节数为实虚部之和，如`c8`=4+4=8字节（对应`complex64`）                    |
| 字符串型           | `np.str_`          | `str_`（固定长度Unicode字符串）    | `U<字符数>`        | 每个字符占4字节（32位Unicode），如`U3`表示3个字符的字符串，总长度3×4=12字节   |
| 日期时间型         | `np.datetime64`    | 按精度划分（年、月、日、时、分、秒） | `M8[单位]`         | 单位包括`Y`（年）、`M`（月）、`D`（日）、`h`（时）、`m`（分）、`s`（秒）等   |

1. 自定义复合类型数组

```python
import numpy as np

data = [
    ('zs', [90, 80, 85], 15),
    ('ls', [92, 81, 83], 16),
    ('ww', [95, 85, 95], 18)
]

# 方式1：
# U3：长度为3的Unicode字符串
# 3int32：3个int32类型数字
# int32：int32类型数字
arr1 = np.array(data, dtype="U3, 3int32, int32")
print(arr1)
print(arr1[0]['f0'])
print(arr1[1]['f1'])
"""
[('zs', [90, 80, 85], 15) ('ls', [92, 81, 83], 16)
 ('ww', [95, 85, 95], 18)]
zs
[92 81 83]
"""

# 方式2：
 # dtype中定义(名称, 类型, 长度)
arr2 = np.array(data, dtype=[('name', 'str_', 2), ('score', 'int32', 3), ('age', 'int32', 1)])
print(arr2)
print(arr2['name']) # 打印name列
print(arr2['score']) # 打印score列
print(arr2['age']) # 打印age列
"""
[('zs', [90, 80, 85], [15]) ('ls', [92, 81, 83], [16])
 ('ww', [95, 85, 95], [18])]
['zs' 'ls' 'ww']
[[90 80 85]
 [92 81 83]
 [95 85 95]]
[[15]
 [16]
 [18]]
"""

# 方式3：
# 名称和类型分开定义
arr3 = np.array(data, dtype={"names":["name", "score", "age"], "formats":["U3", "3int32", "int32"]})
print(arr3[0]['name']) # 打印第一个人的名字
print(arr3[1]['score']) # 打印第二个人的成绩
"""
zs
[92 81 83]
"""

# 方式4：
arr4 = np.array(data, dtype={'names':('U3', 0), 'scores':('3int32', 16), 'age':('int32', 28)})
print(arr4[0]['names']) # 打印第一个人的名字
print(arr4[0]['scores']) # 打印第一个人的成绩
print(arr4.itemsize) # 打印数组中每个元素的字节数
"""
zs
[90 80 85]
32
"""
```
2. 日期类型数组

```python
import numpy as np

f1 = np.array(['2020', '2020-01-01', '2021-01-01 01:01:01', '2020-02-01'])
f2 = f1.astype("M8[D]") # 转换为天数类型
print(f2)
f3 = f2.astype("int32") # 转换为int32类型，表示距离1970年1月1日的天数
print(f3)
f4 = f1.astype("M8[s]").astype("int32")
print(f4)
print(f2[3] - f2[0]) # 计算两个日期的差值
"""
['2020-01-01' '2020-01-01' '2021-01-01' '2020-02-01']
[18262 18262 18628 18293]
[1577836800 1577836800 1609462861 1580515200]
31 days
"""
```
3. 虚数类型数组

```python
import numpy as np

a = np.array([[1 + 1j, 2 + 4j, 3 + 7j],
              [4 + 2j, 5 + 5j, 6 + 8j],
              [7 + 3j, 8 + 6j, 9 + 9j]])
print(a.T) # 转置

for x in a.flat:
    print(x.real, ", ", x.imag)
```
#### （2）维度操作
1. 视图变维（数据共享）：`reshape()`和`ravel()`
2. 就地变维：直接改变原数组对象的维度，不返回新数组
3. 复制变维（数据独立）：`flatten()`

```python
import numpy as np

arr1 = np.arange(1, 9)
print(arr1)
print(arr1.shape)
"""
[1 2 3 4 5 6 7 8]
(8,)
"""

arr2 = arr1.reshape(2, 4)
print(arr2)
print(arr2.shape)
"""
[[1 2 3 4]
 [5 6 7 8]]
(2, 4)
"""

arr3 = arr2.reshape(2, 2, 2)
print(arr3)
print(arr3.shape)
"""
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
(2, 2, 2)
"""

arr4 = arr3.ravel() # 将多维数组转换为一维数组
print(arr4)
"""
[1 2 3 4 5 6 7 8]
"""

arr5 = arr3.flatten() # 将多维数组转换为一维数组，返回一个新的数组
print(arr5)
"""
[1 2 3 4 5 6 7 8]
"""

arr1 += 10
print("arr1：", arr1)
print("arr2：", arr2)
print("arr3：", arr3)
print("arr4：", arr4)
print("arr5：", arr5) # arr5不受arr1影响，因为arr5是arr1的副本
"""
arr1： [11 12 13 14 15 16 17 18]
arr2： [[11 12 13 14]
 [15 16 17 18]]
arr3： [[[11 12]
  [13 14]]

 [[15 16]
  [17 18]]]
arr4： [11 12 13 14 15 16 17 18]
arr5： [1 2 3 4 5 6 7 8]
"""
```
#### （3）切片操作
- `数组对象[起始位置:终止位置:步长]`

```python
import numpy as np

a = np.arange(1, 10)
print(a)  # 1 2 3 4 5 6 7 8 9

print(a[:3])  # 1 2 3
print(a[3:6])   # 4 5 6
print(a[6:])  # 7 8 9
print(a[::-1])  # 9 8 7 6 5 4 3 2 1
print(a[:-4:-1])  # 9 8 7
print(a[-4:-7:-1])  # 6 5 4
print(a[-7::-1])  # 3 2 1
print(a[::])  # 1 2 3 4 5 6 7 8 9
print(a[:])  # 1 2 3 4 5 6 7 8 9
print(a[::3])  # 1 4 7
print(a[1::3])  # 2 5 8
print(a[2::3])  # 3 6 9
```
- `a[:, :, :]`

```python
import numpy as np

a = np.arange(1, 28) # 产生1~28(不包含28)范围内的数字
a.resize(3, 3, 3)
print(a)

# 切出1页
print(a[1, :, :])

# 切出所有页的1行
print(a[:, 1, :])

# 切出0页所有行的第1列
print(a[0, :, 1])
```
- 掩码操作：比作筛子，即把True位置上的元素保留，False位置上的元素丢弃

```python
import numpy as np

arr1 = np.arange(1, 6, 1)
print(arr1)
mask = [True, False, True, False, False] # 掩码数组
print(arr1[mask])
"""
[1 2 3 4 5]
[1 3]
"""

# 实际运用中，mask一般通过计算获取
mask = (arr1 >= 3)
print(mask)
"""
[False False  True  True  True]
"""
```
#### （4）其他属性
- `shape`：维度
- `dtype`：元素类型
- `size`：元素数量
- `ndim`：维数，`len(shape)`
- `itemsize`：元素字节数
- `nbytes`：总字节数
- `real`：复数数组的实部数组
- `imag`：复数数组的虚部数组
- `T`：数组对象的转置视图
- `flat`：扁平迭代器
## 2.2 与`list`互转
- 方式1：利用列表创建数组

```python
import numpy as np

# 创建一个一维 numpy 数组
arr1 = np.array([1, 2, 3, 4, 5])
print(type(arr1))

# 使用 tolist() 方法转换为列表
L = arr1.tolist()
print(type(L))

print(L)  # [1, 2, 3, 4, 5]
```
- 方式2：`asarray()`

```python
# asarray将列表转行为数组
arr2 = np.asarray(L)
print(type(arr2))
print(arr2)
```
- 方式3：列表推导式

```python
import numpy as np

# 创建一个一维 numpy 数组
array_1d = np.array([1, 2, 3, 4, 5])

# 使用列表推导式转换为列表
list_1d = [x for x in array_1d]
print(type(list_1d))
print(list_1d)  # [1, 2, 3, 4, 5]
```

> Tips：
> 1. `ndarray`与`list`比较：
> 
> 		- **归属**：`list` 是 Python 内置类型，`ndarray` 是 `Numpy` 库的类型
> 		- **性能**：在数值计算中，`NumPy` 数组的性能通常优于 Python 列表。因为 `NumPy` 数组是在 C 语言中实现的，而 Python 列表是在 Python 虚拟机中实现的
> 		- **功能**：`Numpy` 数组提供了许多高级的数学函数和操作，而在列表中需要自己手动实现
> 		- **存储类型**：`Numpy` 数组中的所有元素必须是相同的类型，而 Python 列表可以包含不同类型的元素
> 		- **索引和切片**：`Numpy` 数组支持高效的索引和切片操作，而 Python 列表的这些操作通常较慢

-----
==微语录：在逆风中把握方向，做暴风雨中的海燕，做不改颜色的孤星。==
