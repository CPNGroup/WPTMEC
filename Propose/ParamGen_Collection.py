# 这个文件用来计算随机生成的环境状态的标签
import logging
import math
import random
import time

import numpy as np

from Algo2 import Algo2


logging.basicConfig(filename='data/parameters.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def ParamGen_Collection(varsMap):
    with open('data/parameters.log', 'w') as f:
        f.seek(0)
        f.truncate()

    N = varsMap["N"]
    logging.info("N %s", N)

    threshold = 4000  # 设置阈值
    varsMap["threshold"] = threshold
    logging.info("threshold %s", threshold)

    # 随机生成权重向量
    weight_vector = np.random.uniform(low=1, high=101, size=(N,))  # 生成长度为N的随机向量，元素随机分布在0到1之间
    varsMap["weight_vector"] = weight_vector
    logging.info("weight_vector %s", weight_vector)

    d = [random.uniform(2.5, 5) for _ in range(N)]  # 距离列表,原本2.5，5
    varsMap["d"] = d
    logging.info("d %s", d)

    min_f = 0.1e9  # 最小值
    max_f = 0.5e9  # 最大值
    # 生成一个大小为(N,)的随机向量
    f = np.random.uniform(min_f, max_f, size=(N,))  # WSD的算力
    varsMap["f"] = f
    logging.info("f %s", f)

    # 系统相关参数
    p_hap = 3  # HAP功率
    varsMap["p_hap"] = p_hap
    logging.info("p_hap %s", p_hap)

    eta = 0.51  # 能量转化率
    varsMap["eta"] = eta
    logging.info("eta %s", eta)

    de = 2.8  # 路损指数
    varsMap["de"] = de
    logging.info("de %s", de)

    Ad = 411  # 天线增益
    varsMap["Ad"] = Ad
    logging.info("Ad %s", Ad)

    fc = 915e6  # 载波频率
    varsMap["fc"] = fc
    logging.info("fc %s", fc)

    h = [Ad * (3e8 / (4 * np.pi * fc * x)) ** de for x in d]  # 生成向量h
    varsMap["h"] = h
    logging.info("h %s", h)

    B = 2e6  # 信道带宽
    varsMap["B"] = B
    logging.info("B %s", B)

    v = 1e0  # 传输增益
    varsMap["v"] = v
    logging.info("v %s", v)

    p = 0.1  # WSD发射功率
    varsMap["p"] = p
    logging.info("p %s", p)

    N_0 = 1e-10  # 噪声
    varsMap["N_0"] = N_0
    logging.info("N_0 %s", N_0)

    R_u = [v * B * math.log2(1 + p * x / N_0) for x in h]  # 生成上行信道速率
    varsMap["R_u"] = R_u
    logging.info("R_u %s", R_u)

    R_d = [v * B * math.log2(1 + p_hap * x / N_0) for x in h]  # 生成上行信道速率
    varsMap["R_d"] = R_d
    logging.info("R_d %s", R_d)

    phi = 100  # 每比特数据的CPU数
    varsMap["phi"] = phi
    logging.info("phi %s", phi)

    T = 1  # 表示时间帧的长度
    varsMap["T"] = T
    logging.info("T %s", T)

    k = 1e-26  # 本地计算能耗系数
    varsMap["k"] = k
    logging.info("k %s", k)

    f_hap = 1e9  # HAP计算单元的计算频率
    varsMap["f_hap"] = f_hap
    logging.info("f_hap %s", f_hap)

    algo2_target, _, pair_num, _ = Algo2(varsMap)

    return algo2_target, pair_num
