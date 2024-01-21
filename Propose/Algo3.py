import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import logging
from Propose.Structure import Structure


# 先约定已好约定的神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.fc1 = nn.Linear(60, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 11)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 算法3
def Algo3(varsMap):
    start_time = time.perf_counter()  # 捕获算法开始运行时间
    # 加载模型参数
    model = Net()
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()  # 设置模型为评估模式
    # 拿到N的值
    N = varsMap["N"]
    # 外层迭代次数
    f = varsMap["f"]
    h = varsMap["h"]
    weight_vector = varsMap["weight_vector"]
    new_vector = [0] * N * 3  # 用来规范化环境状态的格式
    for i in range(N):
        new_vector[i] = f[i]
        new_vector[N + i] = weight_vector[i]
        new_vector[2 * N + i] = h[i]
    new_vector = np.array(new_vector).astype(np.float32)
    # 加载scaler对象，参数缩放方式，要和训练的时候一样
    with open('model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    file.close()
    # 将新的向量进行归一化
    new_vector = scaler.transform([new_vector])
    # 将归一化后的新向量转换为张量
    new_vector = torch.tensor(new_vector).float()

    # 使用模型进行预测
    with torch.no_grad():
        output = model(new_vector)
        _, predicted_label = torch.max(output.data, 1)
        pairNum = predicted_label.item()  # 拿到预测的最佳匹配对数
        print("Predicted Label:", predicted_label.item())

    DIY_vector = [v1 / (v3 * v2) for v1, v2, v3 in zip(weight_vector, h, f)]

    sorted_indices = np.argsort(DIY_vector) + 1

    varsMap["pairs_number"] = pairNum
    # 按照预测的匹配对数进行构造协作簇
    for i in range(pairNum):
        # 构造新的匹配对
        new_pair = [sorted_indices[-1], sorted_indices[0]]
        # 剔除第一个和最后一个元素,剩下的独立设备索引
        sorted_indices = sorted_indices[1:-1]
        # 更新
        varsMap["pairs_array"].append(new_pair)

    # 根据构建好的协作簇找到独立设备
    numbers = list(range(1, N + 1))
    matched_numbers = set([number for pair in varsMap["pairs_array"] for number in pair])
    unmatched_numbers = list(set(numbers) - matched_numbers)
    unmatched_number = len(unmatched_numbers)
    varsMap["unmatched_numbers"] = unmatched_numbers
    varsMap["unmatched_number"] = unmatched_number

    # 构建完之后导入该方法进行方程的构建,该方法会将最佳值写入到文件的最后一行中
    Structure(varsMap)

    with open('data/best_targets.txt', 'r') as f:
        lines = f.readlines()
    last_line = lines[-1].strip()
    target = -float(last_line)  # 读取最后一行获取最佳值

    end_time = time.perf_counter()
    algo3_time = (end_time - start_time) * 1000
    # 返回目标值和算法运行时间
    return target, algo3_time
