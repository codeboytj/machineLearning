# -*- coding: UTF-8 -*-
import sys
import numpy as np
from sklearn import linear_model

# 一元线性回归
# 参考《机器学习》周志华版，式(3.7)与(3.8)


def fit_from_scikit(sample):
    reg = linear_model.LinearRegression()
    # import pdb;pdb.set_trace()
    reg.fit([[s] for s in sample[:,0]], sample[:,1])
    print('scikit拟合结果%s' % reg.coef_)


def calculate_w(sample):
    # sample需要是numpy的数组
    x_mean=np.mean(sample[:,0])

    # 3个累加和
    sum1=0;
    sum2=0;
    sum3=0;
    for s in sample:
        sum1+=s[1]*(s[0]-x_mean)
        sum2+=s[0]*s[0]
        sum3+=s[0]

    import pdb;pdb.set_trace()
    # 求w
    return sum1/(sum2-(sum3*sum3)/len(sample))


def calculate_b(sample,w):
    # sample需要是numpy的数组

    # 1个累加和
    sum1=0;
    for s in sample:
        sum1+=(s[1]-w*s[0])

    return sum1/np.size(sample)


def fit_from_list(sample):
    # sample=np.array(sample)

    w=calculate_w(sample)
    b=calculate_b(sample,w)

    # import pdb;pdb.set_trace()
    print('拟合结果w=%f，b=%f' % (w,b))
    fit_from_scikit(sample)


def fit_from_file(filename):
    sample = []

    f = open(filename, 'r')
    for line in f.readlines():
        # 把末尾的'\n'删掉
        # 文件中每一行一对数据(x,y)，x与y之间以","分隔
        sample.append(line.strip().split(","))
    f.close()

    sample=np.array(sample).astype(np.float64)
    fit_from_list(sample)


def main(args):
    # args的第一个参数是python程序的文件名：
    # ['linear/yiyuan.py', 'afw', 'wfqw']
    fit_from_file(args[1])


if __name__=="__main__":
    main(sys.argv)
