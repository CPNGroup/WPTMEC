import numpy as np
import logging

# 配置日志记录器
from Propose.Algo1 import Algo1_Run

logging.basicConfig(filename='data/parameters.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


# 该函数根据类别分配结果构造时间分配和任务分配优化问题
def Structure(varsMap):
    # 拿到权重向量
    weight_vector = varsMap["weight_vector"]

    pairs_number = varsMap["pairs_number"]

    pairs_array = varsMap["pairs_array"][:]  # 通过切片防止向量改变

    # 找到独立设备的索引
    unmatched_numbers = varsMap["unmatched_numbers"][:]
    unmatched_number = varsMap["unmatched_number"]

    # 根据以上匹配结果获得优化目标的向量形式
    # 优化变量顺序：a,l
    # a: [a1,a2],l_o:[[l_o^loc,l_o^ap,l_o^p],...],lp:[[l_p^loc],...], lq:[[l_q^loc,l_q^ap],...]
    # 优化变量的个数
    len_X = 2 + pairs_number * 4 + unmatched_number * 2
    logging.info("len_X %s", len_X)

    # 约束条件数量，排除了大于等于0的约束
    len_Constraints = pairs_number + pairs_number + pairs_number + unmatched_number + 1 + pairs_number + pairs_number + \
                      unmatched_number + pairs_number + pairs_number + unmatched_number
    logging.info("len_Constraints(except Non-negative constraints) %s", len_Constraints)

    # 各个变量的大小：
    size_c = len_X + len_Constraints
    size_A = [0, 0]
    size_A[0] = len_Constraints
    size_A[1] = size_c
    size_b = len_Constraints
    logging.info("size_c %s", size_c)
    logging.info("size_A %s", size_A)
    logging.info("size_b %s", size_b)

    # 系统相关参数
    p_hap = varsMap["p_hap"]  # HAP功率
    eta = varsMap["eta"]  # 能量转化率
    h = varsMap["h"]  # 生成向量h
    p = varsMap["p"]  # WSD发射功率
    R_u = varsMap["R_u"]  # 生成上行信道速率
    R_d = varsMap["R_d"]  # 生成上行信道速率
    phi = varsMap["phi"]  # 每比特数据的CPU数
    T = varsMap["T"]  # 表示时间帧的长度
    k = varsMap["k"]  # 本地计算能耗系数
    f = varsMap["f"][:]  # WSD的算力
    f_hap = varsMap["f_hap"]  # HAP计算单元的计算频率
    threshold = varsMap["threshold"]  # 拿到要求处理的数据量最低值

    # 生成向量c,开始是时间分配，随后是已匹配类别，最后是未匹配类别
    c = [0] * size_c
    # 更新匹配对的权重
    for i in range(pairs_number):
        start_index = 2 + i * 4  # 计算切片的起始索引
        end_index = start_index + 4  # 计算切片的结束索引
        c[start_index:end_index] = [-weight_vector[pairs_array[i][0] - 1], -weight_vector[pairs_array[i][0] - 1],
                                    -weight_vector[pairs_array[i][0] - 1],
                                    -weight_vector[pairs_array[i][1] - 1]]  # 匹配对的赋值
    # 更新未匹配对的权重
    for i in range(unmatched_number):
        start_index = 2 + pairs_number * 4 + i * 2
        end_index = start_index + 2  # 计算切片的结束索引
        c[start_index:end_index] = [-weight_vector[unmatched_numbers[i] - 1], -weight_vector[unmatched_numbers[i] - 1]]
    logging.info("c %s", c)

    # 生成向量b
    b = [0] * size_b
    for i in range(pairs_number):
        b[i] = T
        b[pairs_number + i] = T
        b[pairs_number * 5 + unmatched_number * 2 + 1 + i] = -threshold
        b[pairs_number * 6 + unmatched_number * 2 + 1 + i] = -threshold
    for i in range(unmatched_number):
        b[pairs_number * 7 + unmatched_number * 2 + 1 + i] = -threshold
    # HAP时间约束
    b[pairs_number * 3 + unmatched_number] = T
    logging.info("b %s", b)

    # 生成向量A
    # 先将松弛向量的系数赋好值
    A = np.zeros((size_A[0], size_A[1]))
    for i in range(size_A[0]):
        A[i][i - size_A[0]] = 1
    for i in range(pairs_number):
        A[i][0:2] = [T, T]
        A[i][2 + i * 4:2 + i * 4 + 4] = [0, 0, 1 / R_d[pairs_array[i][1] - 1] + phi / f[pairs_array[i][1] - 1], 0]

        A[pairs_number + i][2 + i * 4: 2 + i * 4 + 4] = [0, 0, phi / f[pairs_array[i][1] - 1],
                                                         phi / f[pairs_array[i][1] - 1]]

        A[pairs_number * 2 + i][0:2] = [0, -T]
        A[pairs_number * 2 + i][2 + i * 4: 2 + i * 4 + 4] = [0, 1 / R_u[pairs_array[i][0] - 1],
                                                             1 / R_u[pairs_array[i][0] - 1], 0]

    for i in range(unmatched_number):
        A[pairs_number * 3 + i][2 + pairs_number * 4 + i * 2:2 + pairs_number * 4 + i * 2 + 2] = [0, 1 / R_u[
            unmatched_numbers[i] - 1]]

    A[pairs_number * 3 + unmatched_number][0:2] = [T, T]
    for i in range(pairs_number):
        A[pairs_number * 3 + unmatched_number][2 + i * 4:2 + i * 4 + 4] = [0, phi / f_hap, 0, 0]
    for i in range(unmatched_number):
        A[pairs_number * 3 + unmatched_number][2 + pairs_number * 4 + i * 2:2 + pairs_number * 4 + i * 2 + 2] = [0,
                                                                                                                 phi / f_hap]

    for i in range(pairs_number):
        A[pairs_number * 3 + unmatched_number + 1 + i][0:2] = [-eta * p_hap * h[pairs_array[i][0] - 1] * T, 0]
        A[pairs_number * 3 + unmatched_number + 1 + i][2 + i * 4:2 + i * 4 + 4] \
            = [k * f[pairs_array[i][0] - 1] ** 2 * phi, p / R_u[pairs_array[i][0] - 1], p / R_u[pairs_array[i][0] - 1],
               0]

        A[pairs_number * 3 + unmatched_number + 1 + pairs_number + i][0:2] \
            = [-eta * p_hap * h[pairs_array[i][1] - 1] * T, 0]
        A[pairs_number * 3 + unmatched_number + 1 + pairs_number + i][2 + i * 4:2 + i * 4 + 4] \
            = [0, 0, k * f[pairs_array[i][1] - 1] ** 2 * phi, k * f[pairs_array[i][1] - 1] ** 2 * phi]

    for i in range(unmatched_number):
        A[pairs_number * 3 + unmatched_number + 1 + pairs_number * 2 + i][0:2] \
            = [-eta * p_hap * h[unmatched_numbers[i] - 1] * T, 0]
        A[pairs_number * 3 + unmatched_number + 1 + pairs_number * 2 + i][
        2 + pairs_number * 4 + i * 2:2 + pairs_number * 4 + i * 2 + 2] \
            = [k * f[unmatched_numbers[i] - 1] ** 2 * phi, p / R_u[unmatched_numbers[i] - 1]]

    for i in range(pairs_number):
        A[pairs_number * 5 + unmatched_number * 2 + 1 + i][2 + i * 4:2 + i * 4 + 4] = [-1, -1, -1, 0]
        A[pairs_number * 6 + unmatched_number * 2 + 1 + i][2 + i * 4:2 + i * 4 + 4] = [0, 0, 0, -1]

    for i in range(unmatched_number):
        A[pairs_number * 7 + unmatched_number * 2 + 1 + i][
        2 + pairs_number * 4 + i * 2:2 + pairs_number * 4 + i * 2 + 2] = [-1, -1]
    logging.info("A %s", A)

    # 构造b文件
    filename = "data/b.txt"
    with open(filename, "w") as file:
        # 写入向量长度
        file.write(str(len(b)) + "\n")
        # 写入向量元素
        for element in b:
            file.write(str(element) + "\n")

    # 构造c文件
    filename = "data/c.txt"
    with open(filename, "w") as file:
        # 写入向量长度
        file.write(str(len(c)) + "\n")
        # 写入向量元素
        for element in c:
            file.write(str(element) + "\n")

    # 构造A文件
    filename = "data/A.txt"
    with open(filename, "w") as file:
        # 写入向量行数和列数
        file.write(str(A.shape[0]) + " " + str(A.shape[1]) + "\n")
        # 写入向量元素
        for row in A:
            row_str = " ".join([str(element) for element in row])
            file.write(row_str + "\n")

    # 构造初始点,全都在本地计算：
    x = np.ones(A.shape[1]) * 0.0001
    # 初始化时间分配
    x[0] = 0.5
    x[1] = 0.1
    # 初始化独立设备任务分配
    for i in range(unmatched_number):
        x[2 + pairs_number * 4 + i * 2] = min(eta * p_hap * h[unmatched_numbers[i] - 1] * x[0] * T /
                                              (k * f[unmatched_numbers[i] - 1] ** 2 * phi) / 4,
                                              f[unmatched_numbers[i] - 1] * (1 - x[0]) * T / phi / 4)
        x[2 + pairs_number * 4 + i * 2 + 1] = f_hap / (pairs_number + unmatched_number) * (
                1 - x[0] - x[1]) * T / phi / 4
    # 构造x文件
    for i in range(pairs_number):
        x[2 + i * 4] = min(
            eta * p_hap * h[pairs_array[i][0] - 1] * x[0] * T / (k * f[pairs_array[i][0] - 1] ** 2 * phi),
            f[pairs_array[i][0] - 1] * (1 - x[0]) * T / phi) / 4
        x[2 + i * 4 + 1] = min(
            eta * p_hap * h[pairs_array[i][0] - 1] * x[0] * T / (k * f[pairs_array[i][0] - 1] ** 2 * phi),
            f[pairs_array[i][0] - 1] * (1 - x[0]) * T / phi) / 4
        x[2 + i * 4 + 2] = min(
            eta * p_hap * h[pairs_array[i][1] - 1] * x[0] * T / (k * f[pairs_array[i][1] - 1] ** 2 * phi),
            f[pairs_array[i][1] - 1] * (1 - x[0] - x[1]) * T / phi) / 10
        x[2 + i * 4 + 3] = min(
            eta * p_hap * h[pairs_array[i][1] - 1] * x[0] * T / (k * f[pairs_array[i][1] - 1] ** 2 * phi),
            f[pairs_array[i][1] - 1] * (1 - x[0]) * T / phi) / 4
    for i in range(unmatched_number):
        x[2 + pairs_number * 4 + i * 2] = min(eta * p_hap * h[unmatched_numbers[i] - 1] * x[0] * T /
                                              (k * f[unmatched_numbers[i] - 1] ** 2 * phi),
                                              f[unmatched_numbers[i] - 1] * (1 - x[0]) * T / phi) / 2

    filename = "data/x.txt"
    with open(filename, "w") as file:
        # 写入向量长度
        file.write(str(len(x)) + "\n")
        # 写入向量元素
        for element in x:
            file.write(str(element) + "\n")

    # 将问题，初始点构造好之后就运行算法1进行求解
    Algo1_Run(varsMap)
