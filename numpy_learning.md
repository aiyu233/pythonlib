### numpy learning

#### 1. 创建数组

```python
#input1(数组创建及类名)
import numpy as np
a = np.array([1, 2, 3, 4, 5])
print("a", a, "type:", type(a))
b = np.array(range(1, 6))
print("b", b, "type:", type(b))
#output1
a [1 2 3 4 5] type: <class 'numpy.ndarray'>
b [1 2 3 4 5] type: <class 'numpy.ndarray'>
#input2(数据的类型)
print(a.dtype)
#output2
int32
```

#### 2. numpy中常见数据类型

| 类型                              | 类型代码     | 说明                                                   |
| --------------------------------- | ------------ | ------------------------------------------------------ |
| int8、uint8                       | i1、u1       | 有符号和无符号的8位（1个字节）整型                     |
| int16、uint16                     | i2、u2       | 有符号和无符号的16位（2个字节）整型                    |
| int 32、uint32                    | i4、u4       | 有符号和无符号的32位（4个字节）整型                    |
| int64、uint64                     | i8、u8       | 有符号和无符号的64位（8个字节）整型                    |
| float16                           | f2           | 半精度浮点数                                           |
| float32                           | f4或f        | 标准的单精度浮点数，与C的float兼容                     |
| float64                           | f8或d        | 标准的双精度浮点数，与C的double和Python的float对象兼容 |
| float128                          | f16或g       | 扩展精度浮点数                                         |
| complex64、complex128、complex256 | c8、c16、c32 | 分别用两个32位、64位或128位浮点数表示的复数            |
| bool                              | ?            | 存储True和False值的布尔类型                            |

#### 3. numpy中常见的数组生成函数

##### 3.1 基本函数（array()、arange()、ones()、ones_like()、zeros()、zeros_like()）

|   函数名称   |                 描述                  |
| :----------: | :-----------------------------------: |
|   array()    |      输入一个序列，转换成ndarray      |
|   arange()   |       Python内建range()的数组版       |
|    ones()    |   根据给定形状和数据类型生成全1数组   |
| ones_like()  | 根据所给数组生成一个形状一样的全1数组 |
|   zeros()    |   根据给定形状和数据类型生成全0数组   |
| zeros_like() | 根据所给数组生成一个形状一样的全0数组 |

```python
import numpy as np
#input1:ones()
a = np.ones(5,dtype=np.int)
a = np.ones((5,2))
#output1:
[1 1 1 1 1]
[[1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]]

#input2:zeros()
a = np.zeros(5)
a = np.zeros((5,2))
#output2: 
[0. 0. 0. 0. 0.]
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]

#input3:ones_like() zeros_like()
print(np.zeros_like(a))
print(np.ones_like(a))
#output3:
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
[[1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]
 [1. 1.]]
```

##### 3.2 生成随机数函数

|                      参数                       |                             解释                             |
| :---------------------------------------------: | :----------------------------------------------------------: |
|          np.random.rand(d0,d1,d2,...)           | 创建0-n维(括号里什么都不传就是0维)的均匀分布的随机数数组，浮点数，范围[0,1) |
|          np.random.randn(d0,d1,d2,...)          | 创建0-n维的标准正态分布随机数，浮点数，平均数0，标准差为1，范围(-∞，+∞) |
|   np.random.randint(low,high=None,size=None)    | 离散均匀分布整数,范围是[low,high)；若无high即一个参数，那么范围为[0,low)；size可以是int也可以是tuple，下同 |
| np.random.uniform(low=0.0, high=1.0, size=None) |             产生均匀分布的数组，范围是[low,high)             |
| np.random.normal(loc=0.0, scale=1.0, size=None) |        分布中心为loc，标准差为scale的随机正态分布样本        |
|            np.random.seed(seed=None)            | 如果不传入seed，即括号里不传入参数，那么没作用；如果传入seed，即一个整数，或者是整数序列，那么每次生成的随机数相同 |
|            np.random.permutation(x)             | 输出打乱的序列；如果x是整数，那么输出打乱的[0,x)；如果x是序列，那么返回打乱的序列，如果是多维，则按照最高维度进行打乱 |

###### 3.2.1 np.random.rand()

```python
import numpy as np
#input:np.random.rand(d0,d1,d2,...)
b = np.random.rand(5,2)# 这里传的不是tuple，如果是单个数字，则为一维；若要5行2列，则传入5,2；如果三维，可以传入5,2,3
print(b)
#output
[[0.40179452 0.9836794 ]
 [0.03506931 0.51888164]
 [0.04568291 0.74375556]
 [0.34297351 0.56735105]
 [0.98584524 0.50165574]]
```

###### 3.2.2 np.random.randn()

```python
#input:
a = np.random.randn()
print(a)
#output
-0.11683430647984705
```

###### 3.2.3 np.random.randint()

```python
#input1
a = np.random.randint(3)
print(a)
#output1
1

#input2
a = np.random.randint(-1, 20, (3,2))
print(a)
#output2
[[11 19]
 [ 5  5]
 [-1 19]]
```

###### 3.2.4 np.random.uniform()

```python
#input
a = np.random.uniform()
print(a)
#output
0.8896412820551598
```

###### 3.2.5 np.random.normal()

```python
#input
a = np.random.normal()
print(a)
#output
0.08876541082032546
```

###### 3.2.6 np.random.seed()

```python
#input
for i in range(5):
    np.random.seed([1, 23])
    a = np.random.permutation(10)
    print(a)
#output
[2 6 8 5 3 0 1 9 7 4]
[2 6 8 5 3 0 1 9 7 4]
[2 6 8 5 3 0 1 9 7 4]
[2 6 8 5 3 0 1 9 7 4]
[2 6 8 5 3 0 1 9 7 4]
```

###### 3.2.7 np.random.permutation()

```python
#input1
a = np.arange(6).reshape(2, 3)
print(a)
print(np.random.permutation(a))
#output1
[[0 1 2]
 [3 4 5]]
[[3 4 5]
 [0 1 2]]

#input2
a = np.arange(24).reshape(2, 3, 4)
print(a)
print(np.random.permutation(a))
#output2
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]
 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

[[[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]
 [[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]]
```

#### 4. 数据类型的操作

##### 4.1 指定创建数组的类型(dtype)

```python
#input
a = np.array([1, 0, 1, 0], dtype=bool)   # 用np.bool或是"?"都可以
print(a)
#output
[ True False  True False]
```

##### 4.2 修改数组的数据类型(astype方法)

```python
#input(接4.1的a)
print(a.astype(np.int8))
#outpu
[1 0 1 0]
```

##### 4.3 修改浮点型的小数位数

```python
#input
b =np.array([[0.40179452, 0.9836794],
			 [0.03506931, 0.51888164],
 		     [0.04568291, 0.74375556],
			 [0.34297351, 0.56735105],
			 [0.98584524, 0.50165574]])
print(np.round(b, 2))
#output
[[0.4  0.98]
 [0.04 0.52]
 [0.05 0.74]
 [0.34 0.57]
 [0.99 0.5 ]]
```

#### 5. 数组的形状

##### 5.1 a.reshape()与np.reshape()

```python
#input1
a = np.array([[3,4,5,6,7,8],[4,5,6,7,8,9]])
print(a.shape)
a = a.reshape(3, 4)# tuple和直接数字都可以a = a.reshape((3, 4))
print(a)
#output1
(2, 6)
[[3 4 5 6]
 [7 8 4 5]
 [6 7 8 9]]

#input2: 另一种方式np.reshape
a = np.reshape(a, 12,order="C")# "C"最好加order
print(a)
#output2
[3 4 5 6 7 8 4 5 6 7 8 9]
```

##### 5.2 将数组转化为1维，-1的用法

```python
#input1:
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
length = a.reshape(len(a), -1).shape[0]*a.reshape(len(a), -1).shape[1]
print(a.reshape(length))
#output1
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18]

#input2:-1的用法
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
print(a.reshape(-1,12))
#output2
[[ 1  2  3  4  5  6  7  8  9 10 11 12]]
```

##### 5.3 将数组转化为1维的flatten()方法

```python
#input1
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
a = a.flatten()
print(a)
#output2
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18]
```

#### 6. 数组和数的计算(包括广播机制与轴)

##### 6.1 基本运算

```python
#input
a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18]])
print(a+1)
print(a*2)
b = a + 1
print("a", "\n", a)
print("b", "\n", b)
print("b*a", "\n", b*a)
print("b|a", "\n", b|a)
print("b&a", "\n", b&a)
#output
[[ 2  3  4  5  6  7]
 [ 8  9 10 11 12 13]
 [14 15 16 17 18 19]]
[[ 2  4  6  8 10 12]
 [14 16 18 20 22 24]
 [26 28 30 32 34 36]]
a 
 [[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]
b 
 [[ 2  3  4  5  6  7]
 [ 8  9 10 11 12 13]
 [14 15 16 17 18 19]]
b*a 
 [[  2   6  12  20  30  42]
 [ 56  72  90 110 132 156]
 [182 210 240 272 306 342]]
b|a 
 [[ 3  3  7  5  7  7]
 [15  9 11 11 15 13]
 [15 15 31 17 19 19]]
b&a 
 [[ 0  2  0  4  4  6]
 [ 0  8  8 10  8 12]
 [12 14  0 16 16 18]]
```

##### 6.2 广播机制

如果两个数组的后缘维度（**即从末尾开始算起的维度**）的轴长度相符或其中一方的长度为1，则认为它们是广播兼容的。广播会在缺失和（或）长度为1的维度上进行。

```python
#input
a = np.array([[1,2,3,4,5,6],
              [7,8,9,10,11,12],
              [13,14,15,16,17,18]])
c = np.array([[1],
              [2],
              [3]])
print(c*a)
x = np.array([[1,2,3,4,5,6]])
print(x*a)
k = np.array([1,2,3,4,5,6])
print(k*a)
m = np.array([[[5,6,7,8,9,10]]])
print('m*a', '\n', m*a)
#output
c*a 
 [[ 1  2  3  4  5  6]
 [14 16 18 20 22 24]
 [39 42 45 48 51 54]]
x*a 
 [[  1   4   9  16  25  36]
 [  7  16  27  40  55  72]
 [ 13  28  45  64  85 108]]
k*a 
 [[  1   4   9  16  25  36]
 [  7  16  27  40  55  72]
 [ 13  28  45  64  85 108]]
m*a 
 [[[  5  12  21  32  45  60]
  [ 35  48  63  80  99 120]
  [ 65  84 105 128 153 180]]]
```

##### 6.3 轴的概念

![二维数组轴](https://s2.loli.net/2022/01/13/MuN3JtKqX1gnvRc.jpg)

![三维数组轴](https://s2.loli.net/2022/01/13/wOtCblUdh49JmPq.jpg)

##### 6.4 转置

###### 6.4.1 transpose()方法

```python
#input
t = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18]])
print("t", "\n", t)
print(t.transpose())
#output
t 
 [[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]
[[ 1  7 13]
 [ 2  8 14]
 [ 3  9 15]
 [ 4 10 16]
 [ 5 11 17]
 [ 6 12 18]]
```

###### 6.4.2 swapaxes()(交换轴，只用传入轴)

```python
#input
t = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18]])
print(t.swapaxes(1, 0))
#output
[[ 1  7 13]
 [ 2  8 14]
 [ 3  9 15]
 [ 4 10 16]
 [ 5 11 17]
 [ 6 12 18]]
```

###### 6.4.3 T属性

```python
#input
t = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18]])
print(t.T)
#output
[[ 1  7 13]
 [ 2  8 14]
 [ 3  9 15]
 [ 4 10 16]
 [ 5 11 17]
 [ 6 12 18]]
```

##### 6.5 数组的索引和切片

###### 6.5.1 基本索引与切片

```python
t = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18]])
#input1:一行
print(t[0])
#output1
[1 2 3 4 5 6]

#input2:一列
print(t[:, 2])
#output2
[ 3  9 15]

#input3:多行
print(t[1:3])
#output3
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]

#input4:多列
print(t[:, 2:4])
#output4
[[ 3  4]
 [ 9 10]
 [15 16]]

#input5:某几行
print(t[[1, 2], :])print(t[:, [2, 5]])
#output5
[[ 7  8  9 10 11 12]
 [13 14 15 16 17 18]]

#input6:某几列
print(t[:, [2, 5]])
#output6
[[ 3  6]
 [ 9 12]
 [15 18]]
```

###### 6.5.2 布尔索引

```python
#input
t = np.arange(1,25).reshape(4,6)
print(t < 10)
print(t.all())
#output
[[ True  True  True  True  True  True]
 [ True  True  True False False False]
 [False False False False False False]
 [False False False False False False]]
True
```

##### 6.6 数组的数据修改

###### 6.6.1 普通数值修改

```python
#input
t = np.array([[1, 2, 3, 4, 5, 6],
              [7, 8, 9, 10, 11, 12],
              [13, 14, 15, 16, 17, 18],
              [19, 20, 21, 22, 23, 24]])
t[2, :] = 0
print(t)
#output
[[ 1  2  3  4  5  6]
 [ 7  8  9 10 11 12]
 [ 0  0  0  0  0  0]
 [19 20 21 22 23 24]]
```

###### 6.6.2 利用布尔索引进行修改

```python
#input
t = np.arange(1,25).reshape(4,6)
print(t[t<10])
#output
[1 2 3 4 5 6 7 8 9]
```

###### 6.6.3 利用三元运算符进行修改(np.where)

```python
#input
t = np.arange(1, 25).reshape(4, 6)
print(np.where(t < 10, 0, 10))
#output
[[ 0  0  0  0  0  0]
 [ 0  0  0 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]]
```

##### 6.7 裁剪(clip())

```python
#np.clip(a,b)，小于a的替换a，大于b的替换为b
#input
t = np.arange(1, 25).reshape(4, 6)
print(t.clip(10, 20))
#output 
[[10 10 10 10 10 10]
 [10 10 10 10 11 12]
 [13 14 15 16 17 18]
 [19 20 20 20 20 20]]
```

##### 6.8 数组的拼接

###### 6.8.1 np.vstack()与np.hstack()

```python
#v-vertically h-horizontally
#参数只能是元组
#input
t1 = np.arange(0, 12).reshape(2, 6)
t2 = np.arange(12, 24).reshape(2, 6)
print(np.vstack((t1, t2)))
print(np.hstack((t1, t2)))
#output
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
[[ 0  1  2  3  4  5 12 13 14 15 16 17]
 [ 6  7  8  9 10 11 18 19 20 21 22 23]]
```

###### 6.8.2 np.stack()

```python
# input
t1 = np.arange(0, 12).reshape(2, 6)
t2 = np.arange(12, 24).reshape(2, 6)
print(' t1', '\n', t1, '\n', 't2', '\n', t2)
print('axis=0', '\n', np.stack((t1, t2), axis=0))
print('axis=1', '\n', np.stack((t1, t2), axis=1))
#output
 t1 
 [[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]] 
 t2 
 [[12 13 14 15 16 17]
 [18 19 20 21 22 23]]
axis=0 
 [[[ 0  1  2  3  4  5]
  [ 6  7  8  9 10 11]]
 [[12 13 14 15 16 17]
  [18 19 20 21 22 23]]]
axis=1 
 [[[ 0  1  2  3  4  5]
  [12 13 14 15 16 17]]
 [[ 6  7  8  9 10 11]
  [18 19 20 21 22 23]]]
```

##### 6.9 行(列)互换

```python
#input1:交换行
t = np.arange(0, 24).reshape(4, 6)
print(t)
t[[0, 2], :] = t[[2, 0], :]
print(t)
#output
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
[[12 13 14 15 16 17]
 [ 6  7  8  9 10 11]
 [ 0  1  2  3  4  5]
 [18 19 20 21 22 23]]

#input2:交换列
t = np.arange(0, 24).reshape(4, 6)
print(t)
t[:, [0, 2]] = t[:, [2, 0]]
print(t)
#output
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
[[ 2  1  0  3  4  5]
 [ 8  7  6  9 10 11]
 [14 13 12 15 16 17]
 [20 19 18 21 22 23]]
```

#### 7. numpy中常见的统计函数

| 函数                   | 含义   |
| ---------------------- | ------ |
| t.sum(axis=None)       | 求和   |
| t.mean(a,axis=None)    | 均值   |
| np.median(t,axis=None) | 中值   |
| t.max(axis=None)       | 最大值 |
| t.min(axis=None)       | 最小值 |
| np.ptp(t,axis=None)    | 极差   |
| t.std(axis=None)       | 标准差 |

```python
#input 
t = np.array([[1,23,4,45,6,64,2],
             [2,66,4,6,7,88,1]])
print(np.ptp(t))
print(t.mean(0))
#output
87
[ 1.5 44.5  4.  25.5  6.5 76.   1.5]
```

#### 8.缺失值处理

##### 8.1 nan和inf

```python
#nan:not a number 不是一个数字
#inf：infinity 无穷，inf表示+∞，-inf表示-∞
#######共同点#######
#nan和inf都是float类型
#input
a = np.inf
b = np.nan
print("a", a, "b", b)
print(type(a), type(b))
#output
a inf b nan
<class 'float'> <class 'float'>
```

##### 8.2 nan的特性

```python
#1）两个nan是不相等的
print(np.nan == np.nan)
#output
False  
#2）np.nan!=np.nan
print(np.nan != np.nan)
#output
True
#3）利用以上特性，判断数组中nan的个数
t = np.array([1, 2, 3, 4, np.nan,np.nan])
print((t != t).sum())
print(np.count_nonzero(t != t))
#output
2
2
#4）通过np.isnan(a)来判断，返回bool类型
t[np.isnan(t)] = 0
print(t)
#output
[1. 2. 3. 4. 0. 0.]
#5）nan和任何值运算都为nan
print(np.nan*2)
#output
nan
```

##### 8.3 缺失值(nan)填充

```python
#input 
t = np.arange(0, 24).reshape(4, 6) * 1.0
t[1, 2] = np.nan
t[2, 3] = np.nan
print(t)
for i in range(t.shape[0]):
    nan_num = np.count_nonzero(t[:, i][t[:, i] != t[:, i]])
    if nan_num > 0:
        now_col = t[:, i]
        now_col_not_nan = now_col[np.isnan(now_col) == False].sum()
        now_col_mean = now_col_not_nan / (t.shape[0] - nan_num)
        now_col[np.isnan(now_col)] = now_col_mean
        t[:, i] = now_col
print(t)
#output
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7. nan  9. 10. 11.]
 [12. 13. 14. nan 16. 17.]
 [18. 19. 20. 21. 22. 23.]]
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7. 12.  9. 10. 11.]
 [12. 13. 14. 11. 16. 17.]
 [18. 19. 20. 21. 22. 23.]]
```

#### 9. copy和view

##### 9.1 copy()

```python
#input1:对象引用，a、b完全等价
a = np.arange(0, 24).reshape(4, 6)
b = a
a[0,0] = 999
print(b)
#output1
[[999   1   2   3   4   5]
 [  6   7   8   9  10  11]
 [ 12  13  14  15  16  17]
 [ 18  19  20  21  22  23]]
 
#input2:a=b.copy()，复制，a和b互不影响
a = np.arange(0, 24).reshape(4, 6)
b = a.copy()
a[0, 0] = 999
print(b)
#output
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]]
```

##### 9.2 view()

```python
#input：视图操作，一种切片，会创建新的对象b，但是b的数据完全由a保管，它们的数据变换是一致的
a = np.arange(0, 24).reshape(4, 6)
b = a[:]
a[0, 0] = 999
print(b)
#output
[[999   1   2   3   4   5]
 [  6   7   8   9  10  11]
 [ 12  13  14  15  16  17]
 [ 18  19  20  21  22  23]]
```

*注：黑马程序员学习笔记*

