# encoding: utf-8
from sklearn import linear_model

alpha = 0.01
# 精度设定
epsilon = 1e-8
# 目标函数y=2x+1
x = [1., 2., 3., 4., 5., 6., 7., 8., 9.]
y = [3., 5., 7., 9., 11., 13., 15., 17., 19.]

# scikit-learn的求解方法
def solve_by_scikit():
    # 使用scikit-learn求解
    # reg = linear_model.SGDClassifier(loss="hinge", penalty="l2")
    # reg.fit(x, y)
    print("暂无")


# 采用梯度下降法求解一元线性回归
def solve_by_gradient():
    # 获取循环的长度
    m = len(x)
    a, b, sse2 = 0, 0, 0
    while True:
        grad_a, grad_b = 0, 0
        for i in range(m):
            # 求（a*x(i)+b-y[i])的a,b偏导
            diff = a * x[i] + b - y[i]
            grad_a += x[i] * diff
            grad_b += diff

        grad_a = grad_a / m
        grad_b = grad_b / m

        # 梯形下降(梯形负方向,速度下降最快)迭代求符合最小值的a,b
        # alpha设置迭代步长,即学习率
        a -= alpha * grad_a
        b -= alpha * grad_b

        sse = 0
        for j in range(m):
            sse += (a * x[j] + b - y[j]) ** 2 / (2 * m)
        # 拟合结果判断相差绝对值
        if abs(sse2 - sse) < epsilon:
            break
        else:
            sse2 = sse
    print('{0} * x + {1}'.format(a, b))

def main():
    try:
        print("scikit模拟结果:")
        solve_by_scikit()
        print("梯形下降模拟结果:")
        solve_by_gradient()
    except BaseException as e:
        print("\n=>错误: ", e)


if __name__ == "__main__":
    main()
