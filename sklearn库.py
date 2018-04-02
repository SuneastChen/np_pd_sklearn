# _*_ coding:utf-8 _*_
# !/usr/bin/python

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()  # 加载指定数据库,是一个字典,data与target是key
iris_X = iris.data   # 特征数据表,是二维数组
iris_y = iris.target  # 结果标签,是个一维数组

print(iris_X[:3, :])  # 查看一下三行的数据
print(iris_y)   # 查看结果集

# 将数据集分成训练集,测试集
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print(y_train)   # 训练集自动打乱了

# 用邻近算法
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)  # 开始训练
print(knn.predict(X_test))  # 输入测试集得出结果
print(y_test)   # 这是测试集的真实结果,对比





from sklearn.linear_model import LinearRegression
# 通用的学习模式
loaded_data = datasets.load_boston()  # 加载房价的数据库
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()  # 调用线性回归模式
model.fit(data_X, data_y)  # 训练

print(model.predict(data_X[:4, :]))  # 测试
print(data_y[:4])

print(model.coef_)   # 斜率,即输入特征的各比重
print(model.intercept_)  # 截距
print(model.get_params())  # 返回model定义时的参数
# {'copy_X': True, 'fit_intercept': True, 'n_jobs': 1, 'normalize': False}
print(model.score(data_X, data_y))  # 将数据及结果传入,给线性模型打分,准确度




import matplotlib.pyplot as plt

# 生成数据集X,对应的线性结果集y
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
print(X[:5, :])
plt.scatter(X, y)
plt.show()





from sklearn import preprocessing
a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]])
print(a)
print(preprocessing.scale(a))   # 将各系列的值范围整体缩小





from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2,
                           random_state=22, n_clusters_per_class=1, scale=100)  # 生成数据
# redundant adj.多余的,冗余的  informative adj.提供有用信息的
X = preprocessing.scale(X)  # 坐标轴整体浓缩
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = SVC()   # 加入正则防止过拟合的SVC算法
model.fit(X_train, y_train)
print(model.score(X_test, y_test))  # 浓缩之后得分较高94.4 ,故系列的大小范围直接影响准确度






# 分成好几组的训练集和测试集

from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()  # 加载指定数据库
iris_X = iris.data   # 特征数据表
iris_y = iris.target  # 结果标签表

# 将数据集分成训练集,测试集
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)


knn = KNeighborsClassifier(n_neighbors=5)  # 用邻近算法,加入参数取邻近的5个点

# 只测试一组
# knn.fit(X_train, y_train)  # 开始训练
# print(knn.score(X_test, y_test))   # 只测试一组的结果得分

scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # 分成5组训练集,测试集,分别做测试
print(scores)  # 得到一个一维数组
print(scores.mean())


# 选择最优的参数,即参数取邻近的几个点准确率最高的
k_range = range(1, 31)    # 参数列表
k_scores = []
for k in k_range:  # 也可以把不同的学习model加入测试
    knn = KNeighborsClassifier(n_neighbors=k)  # 加入循环的k参数
    # scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # for classfification(分类问题)
    loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')  # for regression(线性回归问题),加负号
    k_scores.append(loss.mean())  # 每进行一组测试,产生一个一维数组loss

# print(k_scores)

plt.plot(k_range, k_scores)
plt.xlabel('n_neighbors=k')
plt.ylabel('accuracy')
plt.show()

# 得出参数n_neighbors=10时最优,大于时就会产生过度拟合(over fitting)





# 怎么样看过度拟合
'''
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target
train_sizes, train_loss, test_loss = learning_curve(
    SVC(gamma=0.001), X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=[i/10 for i in range(1, 11)]
)   # 多组测试的方法,传入训练数量的百分比点

# print(train_sizes)  # 得到每个时间段训练的数量,组成的一维数组
# print(train_loss)   # 得到相应的二维数组,列数=分组数,行数=时间段的个数
# print(test_loss)    # 得到相应的二维数组,列数=分组数,行数=时间段的个数
train_loss_mean = -np.mean(train_loss, axis=1)  # 在表格右侧求平均,增加列,行不变,即axis=1
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Testing')
plt.xlabel('train_sizes')
plt.ylabel('loss')
plt.show()   # 若将SVC模型的gamma参数改为0.01,便会产生过拟合

'''




# 如何测试模型中的最优参数
'''
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target
param_range = np.logspace(-6, -2.3, 5)   # 新参数
train_loss, test_loss = validation_curve(
    SVC(), X, y, param_name='gamma', param_range=param_range,
    cv=10, scoring='neg_mean_squared_error')   # 返回值无train_sizes,参数无train_sizes,新增了gamma参数

train_loss_mean = -np.mean(train_loss, axis=1)  # 在表格右侧求平均,增加列,行不变,即axis=1
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(param_range, train_loss_mean, 'o-', color='r', label='Training')
plt.plot(param_range, test_loss_mean, 'o-', color='g', label='Testing')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.show()   # 根据图像可直观地看出,最优参数gamma=0.0005左右
'''




# 将训练好的模型,导出导入
from sklearn import svm
iris = datasets.load_iris()
X, y = iris.data, iris.target
model = SVC()
model.fit(X,y)

#方法1:用pickle模块导出导入
import pickle
with open('model.pkl', 'wb')as f:
    pickle.dump(model, f)

with open('model.pkl', 'rb')as f:
    model2 = pickle.load(f)
    print(model2.predict(X[0:3]))  # 把前3行数据做测试


#方法2:用joblib模块,性能更高效
from sklearn.externals import joblib
joblib.dump(model, 'model_joblib.pkl')  # 保存模型

model3 = joblib.load('model_joblib.pkl')
print(model3.predict(X[0:6]))





