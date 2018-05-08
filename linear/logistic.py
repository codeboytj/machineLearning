import random

import numpy as np
from sympy import Symbol
from sympy import exp


class Logistic:
    """
    对数几率回归
    """

    def solve_by_gradient_descent(self, x, y, alpha=0.01, epsilon=1e-6):
        """
        梯度下降法求解。迭代公式推导参考https://blog.csdn.net/javaisnotgood/article/details/78873819
        :param x: 样本数据，x矩阵，为本身样本集合x1的增广矩阵(x1;1)，行数为样本组数，列数为未知数的个数
        :param y: 样本数据，y矩阵
        :param alpha: 学习率
        :param epsilon: 阀值，用来判断迭代的中止条件
        :return: 系数矩阵
        """
        # 1. 先要随机选取最初的系数w0
        (m, n) = x.shape
        w0 = np.ones((1, n))
        w1 = np.ones((1, n))

        tmp = np.zeros((1, m))
        # 迭代计算的中间算子单独拿出来算，不然要重复计算算n遍
        for j in range(m):
            tmp[0,j] = (1 - (1 / (1 + np.exp(w0 * np.transpose(x[j]))))) - y[0,j]

        while True:
            # 2. 根据将系数w0与样本带入迭代公式计算出新的系数w1
            gradient = np.zeros((4,1))

            for i in range(n):
                for j in range(m):
                    # 求梯度
                    gradient[i] += tmp[0,j] * x[j, i]
                w1[0,i] = w0[0,i] - alpha * (gradient[i] / m)

            # 3. 根据w1计算使用新系数时，关于y的损失cost
            cost = np.sum(tmp)
            for j in range(m):
                # 求损失
                tmp[0,j] = (1 - (1 / (1 + np.exp(w1 * np.transpose(x[j]))))) - y[0,j]
            cost1 = np.sum(tmp)

            # 4. 对cost进行评价。如果与上一步的代价之差小于阀值则w0=w1，重复2、3步骤；合适则停止迭代，返回w1作为优化结果
            if np.abs(cost - cost1) < epsilon:
                return w1
            else:
                w0 = w1


def generate_data():
    """
    利用sympy造公式，生成数据
    """

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y = x1 + 3 * x2 + 2 * x3
    z=1.0/(1.0+exp(-y))

    f = open("logistic.csv", "w")

    for i in range(100):
        a=random.uniform(-100,100)
        b=random.uniform(-100,100)
        c=random.uniform(-100,100)
        result = z.evalf(subs={x1: a, x2: b, x3: c})
        # import pdb;pdb.set_trace()
        s = "%f,%f,%f,1,%f\n" % (a, b, c, result)
        f.write(s)

    f.close()


if __name__=="__main__":
    generate_data()

    # 读入数据
    x = np.mat(np.loadtxt("logistic.csv", delimiter=",", usecols=(0, 1, 2)))
    y = np.mat(np.loadtxt("logistic.csv", delimiter=",", usecols=4))

    logistic=Logistic()
    print(logistic.solve_by_gradient_descent(x,y,0.1,1e-6))
