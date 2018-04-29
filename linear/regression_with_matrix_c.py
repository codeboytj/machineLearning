import sys

from sympy import Symbol
import numpy as np
from sklearn import linear_model


# scikit-learn的求解方法
def solve_by_scikit(filename):
    # 读入数据
    x = np.mat(np.loadtxt(filename, delimiter=",", usecols=(0, 1, 2, 3)))
    y = np.mat(np.loadtxt(filename, delimiter=",", usecols=4))

    # 使用scikit-learn求解
    reg = linear_model.LinearRegression()
    reg.fit(x, np.transpose(y))
    print(reg.coef_)


# 采用矩阵运算求解线性回归
def solve_by_matrix(filename):
    # 读入数据
    x = np.mat(np.loadtxt(filename, delimiter=",", usecols=(0, 1, 2, 3)))
    y = np.mat(np.loadtxt(filename, delimiter=",", usecols=4))

    # 计算x的转置xT
    xT = np.transpose(x)

    # 计算xT*x的逆矩阵
    inv_xT_x = np.linalg.inv(xT * x)

    # 求解w
    w = inv_xT_x * xT * np.transpose(y)
    print(w)


def generate_data():
    """
    利用sympy造公式，生成数据
    """

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    x3 = Symbol('x3')
    y = x1 + 3 * x2 + 2 * x3

    f = open("duoyuan.csv", "w")
    for i in range(100):
        result = y.evalf(subs={x1: i, x2: i + 1, x3: i + 2})
        s = "%d,%d,%d,1,%d\n" % (i, i + 1, i + 2, result)
        f.write(s)

    f.close()


def main(args):
    # args的第一个参数是python程序的文件名：
    # ['linear/yiyuan.py', 'afw', 'wfqw']
    solve_by_matrix(args[1])


if __name__ == "__main__":
    main(sys.argv)
