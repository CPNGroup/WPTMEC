import numpy as np
from Propose.utils import printSolutions, read


def Algo1(c, A, b, x0, varsMap):
    N = varsMap["N"]
    min_value = 0
    count = 0
    min_x = []
    solutions = []
    best_solutions = varsMap["best_solutions"]
    targets = []
    best_targets = varsMap["best_targets"]

    if c.shape[0] != A.shape[1]:
        print("A和C形状不匹配")
        return 0
    if b.shape[0] != A.shape[0]:
        print("A和b形状不匹配")
        return 0

    x = x0.reshape(A.shape[1], 1)
    v = np.ones((b.shape[0], 1))
    lam = np.ones((x.shape[0], 1))
    one = np.ones((x.shape[0], 1))
    mu = 1
    n = A.shape[1]
    x_ = np.diag(x.flatten())
    lam_ = np.diag(lam.flatten())
    r1 = np.matmul(A, x) - b
    r2 = np.matmul(np.matmul(x_, lam_), one) - mu * one
    r3 = np.matmul(A.T, v) + c - lam
    r = np.vstack((r1, r2, r3))
    F = r
    n1 = np.linalg.norm(r1)
    n2 = np.linalg.norm(r2)
    n3 = np.linalg.norm(r3)
    zero11 = np.zeros((A.shape[0], x.shape[0]))
    zero12 = np.zeros((A.shape[0], A.shape[0]))
    zero22 = np.zeros((x.shape[0], A.shape[0]))
    zero33 = np.zeros((A.shape[1], A.shape[1]))
    one31 = np.eye(A.shape[1])
    tol = 1e-8
    t = 1
    alpha = 0.5
    while max(n1, n2, n3) > tol:
        print("-----------------step", t, "-----------------")
        # F的Jacobian矩阵
        nablaF = np.vstack((np.hstack((zero11, zero12, A))
                            , np.hstack((x_, zero22, lam_))
                            , np.hstack((-one31, A.T, zero33))))
        # F+nablaF@delta=0,解方程nablaF@delta=-r
        delta = np.linalg.solve(nablaF, -r)  # 解方程，求出delta的值
        delta_lam = delta[0:lam.shape[0], :]
        delta_v = delta[lam.shape[0]:lam.shape[0] + v.shape[0], :]
        delta_x = delta[lam.shape[0] + v.shape[0]:, :]
        # 更新lam、v、x、mu
        alpha = Alpha(c, A, b, lam, v, x, alpha, delta_lam, delta_v, delta_x)
        lam = lam + alpha * delta_lam
        v = v + alpha * delta_v
        x = x + alpha * delta_x
        x_ = np.diag(x.flatten())
        lam_ = np.diag(lam.flatten())
        mu = (0.1 / n) * np.dot(lam.flatten(), x.flatten())
        # 计算更新后的F
        r1 = np.matmul(A, x) - b
        r2 = np.matmul(np.matmul(x_, lam_), one) - mu * one
        r3 = np.matmul(A.T, v) + c - lam
        r = np.vstack((r1, r2, r3))
        F = r
        n1 = np.linalg.norm(r1)
        n2 = np.linalg.norm(r2)
        n3 = np.linalg.norm(r3)
        t = t + 1
        solutions.append(x.flatten())
        print("x的取值", x.flatten())
        print("v的取值", v.flatten())
        print("lam的取值", lam.flatten())
        print("mu的取值", mu)
        print("alpha的取值", alpha)
        z = (c[0:N * 2 + 2].T @ x[0:N * 2 + 2]).flatten()[0] / N  # 求平均值
        targets.append([z])
        print("值为", z)
        if z < min_value:
            # 更新最小值
            min_value = z
            min_x = x.flatten()
            count = 0
        else:
            # 比之前最小值大的次数加1
            count += 1
            if count == 10:
                # 达到停止条件，输出最小值并退出循环
                print(f'已经连续{count}次比之前最小值大，停止迭代，最小值为{min_value}')
                break

    print("##########################找到最优点##########################")
    print("x的取值", min_x)
    printSolutions(solutions, "data/solutions.txt")  # 找打最优解后将迭代过程中的解向量打印到对应文件中
    printSolutions(targets, "data/targets.txt")  # 将每次迭代过程中的目标值打印到对应文件中
    best_targets.append([min_value])
    best_solutions.append(min_x)
    printSolutions(best_targets, "data/best_targets.txt")
    printSolutions(best_solutions, "data/best_solutions.txt")
    print('最优值为', min_value)


# 寻找alpha
def Alpha(c, A, b, lam, v, x, alpha, delta_lam, delta_v, delta_x):
    alpha_x = []
    alpha_lam = []
    for i in range(x.shape[0]):
        if delta_x.flatten()[i] < 0:
            alpha_x.append(x.flatten()[i] / -delta_x.flatten()[i])
        if delta_lam.flatten()[i] < 0:
            alpha_lam.append(lam.flatten()[i] / -delta_lam.flatten()[i])
    if len(alpha_x) == 0 and len(alpha_lam) == 0:
        return alpha
    else:
        alpha_x.append(np.inf)
        alpha_lam.append(np.inf)
        alpha_x = np.array(alpha_x)
        alpha_lam = np.array(alpha_lam)
        alpha_max = min(np.min(alpha_x), np.min(alpha_lam))
        alpha_k = min(1, 0.99 * alpha_max)
    return alpha_k


def Algo1_Run(varsMap):
    A, b, c, x0 = read("data/A.txt", "data/b.txt", "data/c.txt", "data/x.txt")  # 读取问题的构造向量和初始点
    c = c.reshape(-1, 1)
    b = b.reshape(-1, 1)
    Algo1(c, A, b, x0, varsMap)
