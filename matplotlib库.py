# _*_ coding:utf-8 _*_
# !/usr/bin/python


import matplotlib.pyplot as plt
import numpy as np

print('-------------------基本用法-----------------------')
'''
x = np.linspace(-1, 1, 50)
# y = 2*x + 1
y = x**2

plt.plot(x, y)  # 使用plt.plot画(x ,y)曲线
plt.show()  # 使用plt.show显示图像
'''


print('-------------------figure()使用及各项设置-----------------------')
'''
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()  # 使用plt.figure定义一个图像窗口开始,下面都与此图像相关
plt.plot(x, y1)  # 使用plt.plot画(x ,y)曲线
plt.show()  # 使用plt.show显示图像

plt.figure(num=5, figsize=(8, 4))   # 指定figure的num参数,窗口的大小
plt.plot(x, y2, label='up')   # 默认为蓝色的实线
plt.plot(x, y1, color='red', linewidth=2, linestyle='--', label='down')

# plt.plot(传入label参数后)
plt.legend()   # 直接显示图例
# 或用l1,l2接收画的线,指定labels,指定位置loc = 'best'(最佳位置放置)
# plt.legend(handles=[l1, l2], labels=['up', 'down'],  loc='best')

plt.xlim((-1, 2))   # 设置x坐标轴范围：(-1, 2)
plt.ylim((-2, 3))   # 设置y坐标轴范围：(-2, 3)
plt.xlabel('I am x')   # 设置x坐标轴名称：’I am x’
plt.ylabel('I am y')   # 设置y坐标轴名称：’I am y’


x_ticks = np.linspace(-1, 2, 5)
print(x_ticks)
plt.xticks(x_ticks)    # 设置x轴坐标刻度

plt.yticks([-2, -1.8, -1, 1.22, 3],
           ['really bad', 'bad', 'normal', 'good', 'really good'])   # 自定义y轴刻度显示
# 用数学字体显示
# plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])


ax = plt.gca()   # gca = 'get current axis'
ax.spines['right'].set_color('none')  # 不显示右边框
ax.spines['top'].set_color('none')    # 不显示顶边框

# ax.xaxis.set_ticks_position('bottom')   # 设置下边框为x轴
# ax.yaxis.set_ticks_position('left')    # 设置左边框为y轴

ax.spines['bottom'].set_position(('data', 0))  # 将x轴绑定到y轴data=0的位置
ax.spines['left'].set_position(('data', 0))   # 将y轴绑定到x轴data=0的位置
# 或者用(axes,0.5) 绑定在50%的位置
plt.show()

'''

print('-------------------图表注解-----------------------')
'''
x = np.linspace(-3, 3, 50)
y = 2*x + 1

plt.figure(num=1, figsize=(8, 5),)
plt.plot(x, y,)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

x0 = 1
y0 = 2*x0 + 1

plt.scatter([x0, ], [y0, ], s=50, color='b')   # 画点,size=50
plt.plot([x0, x0, ], [0, y0, ], 'k--', linewidth=2.5)   # 画线,(两个点画一条垂直于x轴的虚线)

# 第一种注解方式:
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data',  # 注解的文字,基于哪个点
             xytext=(+30, -30), textcoords='offset points',   # 文字的坐标位置
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle="arc4,rad=.9"))  # 文字的大小及箭头的样式,弧度

# 第二种注解方式:
plt.text(0, 0, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',  # 从文字左下角开始
         fontdict={'size': 16, 'color': 'r'})

plt.show()

'''


print('-------------------tick能见度-----------------------')
'''
x = np.linspace(-3, 3, 50)
y = 0.1*x

plt.figure()
plt.plot(x, y, linewidth=10, alpha=0.5)   # 可以直接设置alpha=0.5
plt.ylim(-2, 2)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
#ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))


# 未实现$$$
# print(ax.get_xticklabels())  # <a list of 9 Text xticklabel objects>
#
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.5))
plt.show()
'''
print('-------------------散点图-----------------------')

'''
n = 1024    # data size
X = np.random.normal(0, 1, n)  # 随机生成1024个平均值为0,方差为1的随机数
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X)  # 根据X,Y点计算出颜色


plt.scatter(X, Y, s=75, c=T, alpha=.5)   # 点的size,点的color,点的alpha
plt.scatter(np.arange(0, 2, 0.2), np.arange(0, 2, 0.2), c='red')

plt.xlim(-1.5, 1.5)
plt.xticks(())  # 设置x轴的刻度,什么参数都不传,就是隐藏
plt.ylim(-1.5, 1.5)
plt.yticks(())

plt.show()
'''


print('-------------------柱形图/条形图-----------------------')
'''
n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)   # 指定范围均匀采样
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

# plt.bar(X, +Y1)   # 画条形图,颜色默认
# plt.bar(X, -Y2)


plt.xlim(-0.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')


# 条形图上显示具体的数值
for x, y in zip(X, Y1):
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')   # 在图表上写文字
    # ha: horizontal alignment(垂直对齐方式)
    # va: vertical alignment(水平对齐方式)

for x, y in zip(X, Y2):
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')



plt.show()
'''

print('-------------------Contours 等高线图-----------------------')
'''
def f(x,y):  # 输入x,y生成高度值
    # the height function
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)   # 将x,y值变成网格平面的X,Y,即栅格化


plt.contourf(X, Y, f(X, Y), 8, alpha=0.75, cmap=plt.cm.cool)
    # 用颜色填充分开的区域 (x, y, z, 0代表分成两部分;8代表分成了10部分;即等高线的密集程度
    #  alpha透明度,cmap即地图的颜色=hot/cool)

C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)    # 画等高线的实线


plt.clabel(C, inline=True, fontsize=10)   # 标注label高度
plt.xticks(())
plt.yticks(())

plt.show()
'''

print('-------------------Image图片-----------------------')

'''
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)


a = np.random.random([3, 4])

plt.imshow(a, interpolation='nearest', cmap='bone', origin='upper')  # 图片展现
#interpolation='nearest'/'none'/'None'/'sinc' 或者用cmap=plt.cmap.bone ,origin='upper'(一致的)/'lower'

plt.colorbar(shrink=0.9)  # colorbar的长度为90%

plt.xticks(())
plt.yticks(())
plt.show()
'''

print('-------------------3D图-----------------------')
'''
from mpl_toolkits.mplot3d import Axes3D

X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)  # height value


fig = plt.figure()
ax = Axes3D(fig)   # 添加三维坐标轴

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
      # 画3D图(x,y,z,rstride(行跨度),cstride(列跨度),或者cmap=plt.get_cmap('rainbow')')

ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
      # 画等高线(zdir从哪个轴方向压下去, offset最低的点偏移量,cmap填充颜色)
ax.set_zlim(-2, 2)   # 设置z轴坐标起始,终点
plt.show()

'''

print('-------------------多图合并显示-----------------------')
'''
# 均匀图中图
plt.figure()

plt.subplot(2, 2, 1)  # 使用plt.subplot来创建小图,取2行2列的第1个位置
plt.plot([0, 1], [0, 1])

plt.subplot(2, 2, 2)
plt.plot([0, 1], [0, 2])

plt.subplot(223)   # 不要","也可以
plt.plot([0, 1], [0, 3])

plt.subplot(224)
plt.plot([0, 1], [0, 4])

plt.show()


# 不均匀图中图,分割的行列数不同
plt.figure()
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

plt.subplot(2, 3, 4)
plt.plot([0, 1], [0, 2])

plt.subplot(235)
plt.plot([0, 1], [0, 3])

plt.subplot(236)
plt.plot([0, 1], [0, 4])

plt.show()  # 展示


# subplot2grid方式:

plt.figure()
#先布局,再画子图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
# (3,3)表示将整个图像窗口分成3行3列,(0,0)表示定位到第0行第0列开始作图，colspan=3表示列的跨度为3
ax1.plot([1, 2], [1, 2])    # 画小图
ax1.set_title('ax1_title')  # 设置小图的标题,加"set_"

# 整体布局设置
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

# 画子图
ax4.scatter([1, 2], [2, 2])
ax4.set_xlabel('ax4_x')
ax4.set_ylabel('ax4_y')

plt.show()


# gridspec方式:

import matplotlib.gridspec as gridspec

plt.figure()
gs = gridspec.GridSpec(3, 3)  # 将整个图像窗口分成3行3列
# 用下标索引的方式指定位置作图
ax6 = plt.subplot(gs[0, :])
ax7 = plt.subplot(gs[1, :2])
ax8 = plt.subplot(gs[1:, 2])
ax9 = plt.subplot(gs[-1, 0])
ax10 = plt.subplot(gs[-1, -2])

plt.show()




# subplots方式:


fig, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex='all', sharey='all')
    # 建立一个2行2列的图像窗口，sharex='all'/'none'表示共享x轴坐标
ax11.scatter([1, 2], [1, 2])
plt.tight_layout()  # 表示紧凑显示图像
plt.show()

'''


print('-------------------图中图-----------------------')
'''
# 创建数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

fig = plt.figure()  # 初始化figure
# 绘制大图
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8    # 左下角的位置以及宽高(百分比)
ax1 = fig.add_axes([left, bottom, width, height])   # fig窗口中添加图
ax1.plot(x, y, 'r')
ax1.set_title('title')

# 绘制小图
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_title('title inside 1')

# 采用一种更简单方法,直接在plt上绘制图
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g')  # 注意对y进行了逆序处理
plt.title('title inside 2')

plt.show()
'''

print('-------------------次坐标轴-----------------------')
'''
x = np.arange(0, 10, 0.1)
y1 = 0.05 * x**2
y2 = np.sin(x)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # 生成ax1的如同镜面效果-->ax2

ax1.plot(x, y1, 'g-')   # green, solid line
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')

ax2.plot(x, y2, 'b--')  # blue
ax2.set_ylabel('Y2 data', color='b')

plt.show()

'''

print('-------------------动画-----------------------')

from matplotlib import animation

x = np.arange(0, 2*np.pi, 0.01)
y = np.sin(x)

fig, ax = plt.subplots()

line, = ax.plot(x, y)   # 在ax图上画线

def animate(i):   # 构造自定义动画函数,参数表示第i帧,这个函数是关键
    line.set_ydata(np.sin(x + i/10))
    return line,


def init():   # 构造开始帧函数init
    line.set_ydata(np.sin(x))
    return line,


# 调用FuncAnimation函数生成动画
ani = animation.FuncAnimation(fig=fig,
                              func=animate,  # 自定义动画函数，即传入刚定义的函数animate
                              frames=100,   # 动画长度，一次循环包含的帧数
                              init_func=init,  # 自定义开始帧，即传入刚定义的函数init
                              interval=20,   # 更新频率，以ms计
                              blit=False)  # 选择更新所有点，还是仅更新产生变化的点,应选择True

plt.show()