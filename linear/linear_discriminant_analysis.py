import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class LinearDiscriminantAnalysis:
    """
    线性判别分析
    """

    def fit(self, X, y):
        """
        :param X: 样本，(m,n)矩阵
        :param y: 样本的类别，m向量。二分类
        :return: 直线系数向量
        """
        # 两种类别分开
        X1 = [[],[]]
        X0 = [[],[]]
        for k, v in enumerate(y):
            if v == 0:
                X0[0].append(X[0][k])
                X0[1].append(X[1][k])
            elif v == 1:
                X1[0].append(X[0][k])
                X1[1].append(X[1][k])

        # 均值向量
        u0 = np.mean(X0, axis=1)
        u1 = np.mean(X1, axis=1)

        # 协方差矩阵
        cov0 = np.cov(X0)
        cov1 = np.cov(X1)

        sw = cov0 + cov1

        w = np.dot(np.linalg.inv(sw), (u0 - u1))

        return w

if __name__=="__main__":

    # 读入数据
    lda=LinearDiscriminantAnalysis()
    print(lda.fit([[1,4,5,-2,-5,-2],[1,3,6,-2,-6,-1]],[0,0,0,1,1,1]))

    print(LDA(solver='eigen').fit([[1,1],[4,3],[5,6],[-2,-2],[-5,-6],[-2,-1]],[0,0,0,1,1,1]).coef_)
    #
