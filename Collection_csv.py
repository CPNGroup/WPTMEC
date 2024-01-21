import pandas as pd

from Propose.ParamGen_Collection import ParamGen_Collection

M = 10000
for i in range(M):
    N = 20
    varsMap = {"N": N}
    _, pairNum = ParamGen_Collection(varsMap)
    sample = {}
    for l in range(N):
        key = "f" + str(l)
        sample[key] = varsMap["f"][l]
        key = "h" + str(l)
        sample[key] = varsMap["h"][l]
        key = "w" + str(l)
        sample[key] = varsMap["weight_vector"][l]
        sample["label"] = pairNum

    # print(sample)

    # 读取CSV文件
    df = pd.read_csv('data/N=20F=2.csv')
    # 遍历字典向量并将值打印在相应标题下

    # 将字典的值添加到相应标题下
    row = []
    for col in df.columns:
        row.append(sample[col])
    df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

    # 将更新后的数据保存回CSV文件
    df.to_csv('data/N=20F=2.csv', index=False)