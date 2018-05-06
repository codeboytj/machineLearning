# 线性回归

线性回归的问题是找到*w*，使得拟合曲线最接近实际趋势，而评价方法是最小二乘法。而求解最小二乘可以采用矩阵运算以及梯度下降两种方法。

## 矩阵运算法

参考式(3-11)，矩阵的转置是很好求的，于是求解最小二乘的难点就落在了求矩阵的逆矩阵之上．而求解逆矩阵是很难的，有很多方法，可以参考这个[帖子](https://www.zhihu.com/question/19584577)

## 梯度下降法

### 求解一元线性

>梯度下降求解算法是一种迭代算法,即在求最小二乘的时候
朝向梯形负方向(梯形正方向定义为增长速度最快)按照一定步长迭代下降
步长也就是定义的学习率

![偏导公式结果](https://s1.ax1x.com/2018/05/04/CNpi40.png)

### 求解多元

参考文章:[梯度下降从放弃到入门](https://yq.aliyun.com/articles/232003?spm=a2c4e.11153940.blogcont541976.17.1e0b87a8LtU5Ti)
[梯度下降小结](https://www.cnblogs.com/pinard/p/5970503.html)