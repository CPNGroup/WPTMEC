import time

import numpy as np
import random
import logging
from Propose.Structure import Structure


def Algo2(varsMap):
    start_time = time.perf_counter()
    # 拿到N的值
    N = varsMap["N"]
    # 外层迭代次数
    p = 0
    # 外层确定的匹配结果
    pairMap = {}
    # 前p次迭代拿到的解
    best_solutions = []
    best_targets = []

    # 添加新的key-value
    varsMap["best_solutions"] = best_solutions
    varsMap["best_targets"] = best_targets

    # 匹配对数初始化
    pairs_number = 0
    varsMap["pairs_number"] = pairs_number

    # 设备索引
    numbers = list(range(1, N + 1))

    # 初始匹配结果
    pairs_array = []
    varsMap["pairs_array"] = pairs_array

    # 找到独立设备的索引
    matched_numbers = set([number for pair in pairs_array for number in pair])
    unmatched_numbers = list(set(numbers) - matched_numbers)
    unmatched_number = len(unmatched_numbers)
    varsMap["unmatched_numbers"] = unmatched_numbers
    varsMap["unmatched_number"] = unmatched_number

    pairMap["match_" + p.__str__()] = pairs_array.copy()
    pairMap["unmatch_" + p.__str__()] = unmatched_numbers.copy()

    Structure(varsMap)

    # 初始化最佳值
    best_target = 0

    # 读取上一次最佳值
    with open('data/best_targets.txt', 'r') as file:
        lines = file.readlines()
    last_line = lines[-1].strip()
    last_target = float(last_line)

    # 拿到全部都是独立设备的最佳结果
    with open('data/best_solutions.txt', 'r') as file:
        lines = file.readlines()
    first_line = lines[0].strip()
    vector_str = first_line.split()
    Initial_Solution = [float(x) for x in vector_str]

    # 拿到初始卸载向量
    Initial_offload = [0] * N
    Initial_local = [0] * N
    for i in range(N):
        Initial_offload[i] = Initial_Solution[2 + i * 2 + 1]
        Initial_local[i] = Initial_Solution[2 + i * 2]

    weight_vector = varsMap["weight_vector"]
    h = varsMap["h"]
    f = varsMap["f"]
    # Offload_Local = [v1 + v2 for v1, v2 in zip(Initial_offload, Initial_local)]
    DIY_vector = [v1/(v3 * v2) for v1, v2, v3 in zip(weight_vector, h, f)]
    # DIY_vector = [v1 * v2 for v1, v2 in zip(Initial_offload, Initial_local)]
    # 对向量进行排序
    # sorted_indices = np.argsort(Offload_Local) + 1
    sorted_indices = np.argsort(DIY_vector) + 1
    # 输出排序后的索引向量
    print("sorted_indices", sorted_indices)
    # 初始化最佳匹配结果
    best_pair = []

    while len(sorted_indices) > 1:
        if last_target < best_target:
            best_target = last_target
            best_pair = pairMap["match_" + p.__str__()]

        p = p + 1
        # 构造新的匹配对
        new_pair = [sorted_indices[-1], sorted_indices[0]]
        # 剔除第一个和最后一个元素,剩下的独立设备索引
        sorted_indices = sorted_indices[1:-1]
        print("new_pair", new_pair)

        # 更新
        pairs_number = pairs_number + 1
        varsMap["pairs_number"] += 1
        varsMap["pairs_array"].append(new_pair)

        # 找到独立设备的索引
        matched_numbers = set([number for pair in varsMap["pairs_array"] for number in pair])
        unmatched_numbers = list(set(numbers) - matched_numbers)
        unmatched_number = len(unmatched_numbers)
        varsMap["unmatched_numbers"] = unmatched_numbers
        varsMap["unmatched_number"] = unmatched_number

        pairMap["match_" + p.__str__()] = varsMap["pairs_array"].copy()
        pairMap["unmatch_" + p.__str__()] = unmatched_numbers.copy()

        Structure(varsMap)

        with open('data/best_targets.txt', 'r') as f:
            lines = f.readlines()
        last_line = lines[-1].strip()
        last_target = float(last_line)

    if last_target < best_target:
        best_target = last_target
        best_pair = pairMap["match_" + p.__str__()]

    print("best_target", best_target)
    print("best_pair", best_pair)
    print("best_pair_number", len(best_pair))

    end_time = time.perf_counter()
    algo2_time = (end_time - start_time) * 1000

    # 写入每次迭代过程中选中匹配对的变化
    with open("data/match.txt", "w") as file:
        # 遍历字典中的键值对
        for key, value in pairMap.items():
            # 将键值对写入文件中
            file.write(f"{key}: {value}\n")
    algo2_target = -best_target
    pair_num = len(best_pair)
    return algo2_target, algo2_time, pair_num
