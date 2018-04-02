# _*_ coding:utf-8 _*_
# !/usr/bin/python

import pandas as pd
import numpy as np


print('---------------数据表的创建-------------------')
s = pd.Series([1, 5, np.nan, 44, 9])
print(s)   # 生成了一个系列的值,自动添加了下标索引

dates = pd.date_range('20170101', periods=6)
print(dates)   # 生成了6个连续的日期的索引 DatetimeIndex

df = pd.DataFrame(np.random.rand(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)   # 主体数据为6行4列的随机数,行索引为dates,指定列索引

df1 = pd.DataFrame(np.arange(12).reshape(3, 4))
print(df1)   # 不指定行列索引时,默认从0开始

df2 = pd.DataFrame({'a': s, 'b': np.array([5, 9, -3, -7, 8]), 'c': '中国人'})
print(df2)   # 用字典的形式创建数据表

print(df2.dtypes)   # 查看每一列的数据类型
print(df2.index)    # 查看记录的索引
print(df2.columns)  # 查看列的索引,即列名称
print(df2.values)   # 查看数据组成的二维数组
print(df2.describe())   # 将数值类型的列作各种统计
print(df2.T)   # 将表转置

print(df2.sort_index(axis=1, ascending=False))  # 行不变,整列移动,进行索引排序
print(df2.sort_index(axis=0, ascending=False))  # 列不变,整行移动,进行索引排序
print(df2.sort_values(by='b', ascending=False))  # 按b列的值进行降序排序(默认True是升序)
    


print('---------------选择数据-------------------')
dates = pd.date_range('20170101', periods=6)
df3 = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
print(df3)
print(df3['A'])  # 打印A列的数据,同时有行的索引
print(df3.A)  # 打印A列的数据,同上

print(df3[0:3])    # 根据行的下标取数据
print(df3['20170101':'20170103'])  #根据行的索引值取数据

# 通过loc的标签索引值来选取
print(df3.loc['20170104'])   # 根据行的具体索引值取数据
print(df3.loc[:, ['A', 'B']])  # 取 所有行数据,AB列数据
print(df3.loc['20170104', ['A', 'B']])  # 取 20170104的行数据,AB的列数据

# 通过iloc的行列下标选取
print(df3.iloc[3])  # 取第4行的数据
print(df3.iloc[3, 1])  # 取第4行,第2列的数据
print(df3.iloc[2:5, 0:3])   # 取部分行,部分列数据
print(df3.iloc[1:6][0:3])  # 这个是根据行筛选了两遍
print(df3.iloc[[1, 3, 5], 1:3])  # 取不连续行,部分列数据


# 以上两种方式混合,ix
print(df3.ix[:3, ['A', 'C']])

# 根据某列的条件筛选选择,适用 &,|,!
print(df3[(df3.A > 9) & (df3.A < 18)])


print('---------------设置值-------------------')
df3.iloc[2, 2] = 999
df3.loc['20170101', 'B'] = 888
df3.ix[1, 'D'] = 777

# 常用的方法
df3.B[df3.A > 15] = 6789     # 将A列大于15的记录选出,将B列值改为6789
df3.A[df3.A > 15] = 0     # 将A列大于15的值改为0

print(df3)

print('---------------增加列,增加行-------------------')
df3['F'] = np.nan  # 增加一个空列F
df3['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20170101', periods=6))  # 增加一个系列E的值
print(df3)

s1 = pd.Series([11, 22, 32, 44, 55, 66], index=['A', 'B', 'C', 'D', 'E', 'F'])
res = df3.append(s1, ignore_index=True)    # 增一条记录,必须要忽略掉索引参数
print(res)


print('---------------处理数据丢失-------------------')
df3 = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A', 'B', 'C', 'D'])
df3.iloc[0, 1] = np.nan
df3.iloc[1, 2] = np.nan
print(df3.dropna(axis=0, how='any'))  # 列不变,丢弃行,如果有任何nan值,默认为'any',可以设为'all'

print(df3.fillna(value=0))   # 如果数据为null,以设置的值填充
print(np.any(df3.isnull()) == True)  # 判断数据表中有无nan值


print('---------------数据导入导出-------------------')
data = pd.read_excel('students.xls')
print(data)    # 默认会给行加下标索引

mydata = data.pivot_table(index='sex', values='age', aggfunc=np.mean)   # 生成数据透视表
print(mydata)

df3.to_csv('df3.csv')   # 数据保存
data = pd.read_csv('df3.csv')
print(data)

# 常用类型read_csv,read_excel,read_sql,read_json,read_html,read_pickle


print('---------------数据表的合并-------------------')

print('---------------concat合并')
# 多张表列完全相同时的合并
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])

res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)  # 保持列不变,行合并,忽略掉索引重新建
print(res)
res = df1.append([df2, df3], ignore_index=True)    # 尾部批量追加行合并
print(res)


# 多张表,部分列相同时的合并
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])

# 1.相当于mysql的union,往下合并
res = pd.concat([df1, df2], join='outer', ignore_index=True)  # 默认outer,显示全部列,没有的字段显示NaN
print(res)

res = pd.concat([df1, df2], join='inner', ignore_index=True)  # 设为inner,不显示有NaN的字段
print(res)

# 2.axis=1 左右合并
# 相当于mysql的左连接(两个表的列全部显示)
res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])
print(res)

# 相当于全连接(两个表的列全部显示), 不加join_axes参数
res = pd.concat([df1, df2], axis=1)
print(res)


print('---------------merge合并')
sex_data = pd.DataFrame({'name': ['z3', 'l4', 'w5', 'z6'],
                         'sex': [0, 1, 1, 0]})
age_data = pd.DataFrame({'name': ['l4', 'w5', 'z6', 's7'],
                         'age': [24, 26, 21, 29]})

print(sex_data)
print(age_data)

# 内联接,相同的'name'列,只显示一次
res = pd.merge(sex_data, age_data, on='name')  # 默认how='inner'
print(res)    # on=['列名1','列名2']也可以根据多个列来合并

# how = 'inner','outer','left','right'

res = pd.merge(sex_data, age_data, on='name', how='outer', indicator=True)  # indicator=True,多一个列,显示哪个表有数据
print(res)      # indicator='where',列名显示'where'

# 当两张表连接时,同时存在相同的字段时,无法区分到底是哪张表的,这时可加入参数,suffixes=['_左表名','_右表名']

# 创建有索引的两张表,通过索引来merge
sex_data = pd.DataFrame({'sex': [0, 1, 1, 0]}, index=['z3', 'l4', 'w5', 'z6'])
age_data = pd.DataFrame({'age': [24, 26, 21, 29]}, index=['l4', 'w5', 'z6', 's7'])

print(sex_data)
print(age_data)

res = pd.merge(sex_data, age_data, left_index=True, right_index=True, how='outer')
print(res)


print('---------------plot图表-------------------')
import matplotlib.pyplot as plt
data = pd.Series(np.random.randn(1000), index=np.arange(1000))  # index默认就是np.arange(1000)
data = data.cumsum()   # 进行数据累加求和,成为一维数组

data.plot()
plt.show()


data = pd.DataFrame(np.random.randn(1000, 4),
                    index=np.arange(1000),
                    columns=list('ABCD'))
data = data.cumsum()
print(data.head())   # 查看前5行的数据
data.plot()
plt.show()     #绘制4个系列的数据

#plot的方法:'bar','hist','box','kde','area','scatter','hexbin','pie'
# 绘制一个系列的散点图
data.plot.scatter(x='A', y='B')  # 散点图
plt.show()


# 绘制二个系列的散点图
ax = data.plot.scatter(x='A', y='B', label='A_B', color='DarkBlue')
data.plot.scatter(x='A', y='C', label='A_C', color='Darkgreen', ax=ax)
plt.show()








