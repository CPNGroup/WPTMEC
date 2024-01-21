from Propose.ParamGen import *


# 记录各种算法获得的目标
algo2_Sum,  algo3_sum = 0, 0
# 记录各种算法的运行时间
time_algo2_Sum, time_algo3_Sum = 0, 0

# 循环次数，蒙特卡洛实验
M = 200
for i in range(M):
    N = 20  # 设置终端设备数量
    varsMap = {"N": N}

    threshold = 4000  # 设置阈值
    varsMap["threshold"] = threshold

    p_hap = 3  # 设置HAP功率
    varsMap["p_hap"] = p_hap

    eta = 0.51  # 能量转化率
    varsMap["eta"] = eta

    de = 2.8  # 路损指数
    varsMap["de"] = de

    Ad = 411  # 天线增益
    varsMap["Ad"] = Ad

    fc = 915e6  # 载波频率
    varsMap["fc"] = fc

    B = 2e6  # 信道带宽
    varsMap["B"] = B

    v = 1e0  # 传输增益
    varsMap["v"] = v

    p = 0.1  # WSD发射功率
    varsMap["p"] = p

    N_0 = 1e-10  # 噪声
    varsMap["N_0"] = N_0

    phi = 100  # 每比特数据的CPU数
    varsMap["phi"] = phi

    T = 1  # 表示时间帧的长度
    varsMap["T"] = T

    k = 1e-26  # 本地计算能耗系数
    varsMap["k"] = k

    f_hap = 1e9  # HAP计算单元的计算频率
    varsMap["f_hap"] = f_hap

    algo2_target, algo3_target, algo2_time, algo3_time = ParamGen(varsMap)
    algo2_Sum += algo2_target
    algo3_sum += algo3_target
    time_algo2_Sum += algo2_time
    time_algo3_Sum += algo3_time

print("所提方案目标", algo2_Sum / M)
print("所提方案2目标", algo3_sum / M)
print("所提方案时间", time_algo2_Sum / M)
print("所提方案2时间", time_algo3_Sum / M)
