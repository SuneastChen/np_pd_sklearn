# _*_ coding:utf-8 _*_
# !/usr/bin/python

import numpy as np

print(np.version)    # 能输出包的路径
print(np.__file__)   # 纯粹输出包的路径
print(np.version.version)  # 输出版本号,1.21.1

print('---------------生成数组-------------------')
# 使用numpy.array方法产生一维数组,就是列表中无逗号
a1 = np.array([1, 2, 3, 4])  # 接受一个列表或元组;(元素应为同一类型的,不是的话,np会进行转换)
print(a1)
print(type(a1))    # <class 'numpy.ndarray'>

# 生成数组的时候，可以指定数据类型，例如numpy.int32, numpy.int16, and numpy.float64等
a3 = np.array((1.2, 2, 3, 4), dtype=np.int32)
print(a3)   # ---> [1 2 3 4]

# 使用numpy.array方法产生二维数组
a2 = np.array([[1, 2, 3], [3, 4, 5]])  # 接受一个列表组成的列表,即产生二维数组
print(a2)



# 创建一个三维数组
a5 = np.zeros((2, 2, 2))  # 即两个两行两列的数组
print(a5)
# ---> 输出如下:
# [[[ 0.  0.]
#   [ 0.  0.]]
#
#  [[ 0.  0.]
#   [ 0.  0.]]]

# 一维数组打印成行， 二维数组打印成矩阵。 三维数组打印成矩阵列表。



print('---------------数组的属性-------------------')
print(a2.ndim)  # 数组的维数,数组的秩，也称为轴---> 2
print(a2.shape)  # 数组每一维度的数量,shape是一个元组---> (2, 3),即0个2行3列的数组
print(a2.size)  # 数组的元素个数 ---> 6
print(a2.dtype)  # 元素的类型 ---> int32
print(a2.itemsize)  # 每个元素所占的字节数 ---> 4



print('---------------构造特定的矩阵-------------------')
print(np.zeros((3, 4)))  # 必须接受一个元组或列表
print(np.ones((3, 4)))
print(np.eye(5))  # 产生5行5列的对角线为1,其他为0的矩阵

# 使用numpy.arange构造有序的一维数组,参数同range(起始,终止,步长),指定默认步长为1
a4 = np.arange(15)
print(a4)
print(a4.reshape((3, 5)))  # 变换为3行5列的矩阵

# 使用numpy.linspace方法创建有序的一维数组,指定个数来创建
print(np.linspace(1, 4, 9))  # 在从1到4中平均分配产生9个数的一维数组,默认是包括未尾的4
print(np.linspace(1, 4, 9, endpoint=False).reshape((3, 3)))  # 可以指定不包括末尾




print('---------------numpy中的函数-------------------')
a9 = np.random.random((3, 4))
print(a9)   # 产生2行4列的随机矩阵

print(np.max(a9))
print(np.min(a9))
print(np.sum(a9))
print(np.mean(a9))
print(np.average(a9))
print(np.median(a9))

# np.函数的axis通用参数
print(np.sum(a9, axis=0))  # 哪个轴的数量不变,在另一轴上操作
print(np.max(a9, axis=1))


print(np.square(a9))  # 各元素求平方
print(np.sin(a9))  # 求三角函数
print(a9)  # floor型的
print(np.int8(a9))  # int型的


b2 = np.arange(2, 14).reshape((3, 4))
print(b2)
print(np.argmin(b2))   # 求最小数字的索引 ---> 0
print(np.argmax(b2))   # 求最大数字的索引 ---> 11
print(np.cumsum(b2))   # 累加求和成为一维数组
print(np.diff(b2))     # 返回元素之间的相差,行数不变,列数少1
print(np.nonzero(b2))  # 返回所有非0元素的 [行数的数组] 与 [列数的数组]
print(np.sort(b2))     # 默认每一行进行排序,或指定axis参数
print(np.transpose(b2))  # 矩阵的转置
print(np.clip(b2, 5, 9))  # 小于5的数变成5,大于9的数变成9



print('---------------numpy中索引和切片-------------------')
# numpy的切片,由外到内一层层切片
a6 = np.array([[2, 3, 4], [5, 6, 7]])
print(a6[1])  # ---> [5, 6, 7]
print(a6[1, 2])  # ---> 7
print(a6[1][2])# python的表示也可以

print(a6[1, :])  # 第2行的所有列的值---> [5 6 7]
print(a6[:, 1])  # 所有行的第2列的值---> [3 6]
print(a6[1, 1:2])  # 第2行的,第[m,n)列的值 ---> [6]   1:2同样是不包括下标2的
a6[1, :] = [8, 9, 10]
print(a6)   # ---> 将[5 6 7] 改为 [8 9 10]


# 使用for操作元素
b2 = np.arange(2, 14).reshape((3, 4))
for row in b2:  # 迭代行
    print(row)

for column in b2.T:  # 转置后迭代行,即迭代了列
    print(column)

print(b2.flatten())  # 返回一维数组
for item in b2.flat:   # 迭代元素内容
    print(item)



print('---------------数组基本的数据运算-------------------')
a7 = np.ones((2, 2))
a8 = np.eye(2)

print(a6 > 3)   # 给定一个条件,元素依次判断,返回布尔值相对应的的矩阵
print(a6[(a6 < 3) | (a6 > 8)])   # 给定条件,取数据,成一维数组,&,|,!
print(a7 + a8)  # 矩阵的加法,每个元素对应位置相加
print(a7 - a8)
print(a8 * 2)  # 矩阵的数乘,每个元素与数相乘
print((a8 * 2)**4)  # 矩阵的求幂
print(a7 * a8)  # 对应的位置直接相乘
print(a8/(a7*2))  # 对应的位置直接相乘除


print('---------------数组自带的方法-------------------')
a9 = np.array([[2, 3, 4], [5, 6, 7]])
print(a9.sum())   # 元素求和
print(a9.min())   # 求最小的元素 ---> 2
print(a9.max())   # 求最大的元素 ---> 7
print(a9.mean())   # 求平均值

# 或加入axis参数
print(a9.sum(axis=0))  # 列数量保持不变,增加计算的行,来求每一列的和 ---> [ 7  9 11]
z = a9.astype(np.float64)  # 转换类型
print(z)

# reshape()  数据的变换


print('---------------矩阵的运算-------------------')
a7 = np.ones((2, 2))
a8 = np.eye(2)
# 结果矩阵第m行与第n列交叉位置的那个值，等于第一个矩阵第m行与第二个矩阵第n列，对应位置的每个值的乘积之和。
print(np.dot(a7, a8))  # 这才是矩阵的乘法
print(a7.dot(a8))    # 矩阵有dot()方法

print('--------------矩阵的转置')
print(a9.T)  # 矩阵的转置属性
print(a9.transpose())
print(a9.swapaxes(0, 1))  # 轴交换,即维度交换,行变列,列变行,同转置

print('-------------一维数组的转置')
a15 = np.array([1, 2, 3, 4])
print(a15.T)  # 或transpose()都无法转置,因为只是一维的 ---> [1 2 3 4]

print(a15.reshape(4, 1))   # 指定行数列数转换
print(a15[:, np.newaxis])   # 行数照抄,新增一个列维度

print(a15[:, np.newaxis].shape)  # ---> (4, 1)
print(a15.shape)  # ---> (4,)  一维的数组形式


print('-------------数组的合并')
a10 = np.vstack((a7, a8, a7))  # 垂直合并数组,前提是列数需相同
print(a10)
a11 = np.hstack((a7, a8))  # 水平合并数组,前提是行数数需相同
print(a11)
b3 = np.concatenate((a7, a8, a7), axis=0)  # 保持y轴数量不变,对行进行合并
print(b3)


# 合并数组在内存中重新将结果给指定变量,不存在浅拷贝
a12 = a9    # 浅拷贝,同python
print(a12 is a9)   # True
a13 = a9.copy()
print(a13 is a9) # 深拷贝


print('-------------数组的分割')
a14 = np.arange(12).reshape((3, 4))
print(np.split(a14, 2, axis=1))  # 保持行数不变,对列进行等量分割成2块 (分成的块数,要能整除,即等量分割)
print(np.array_split(a14, 3, axis=1))  # 对列进行不等量分割成3块

print(np.vsplit(a14, 3))    # 垂直方向上砍成等量的3块
print(np.hsplit(a14, 2))   # 水平方向上砍成等量的2块数组


