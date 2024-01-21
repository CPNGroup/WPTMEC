# solutions是一个列表，每个元素是一个向量，该向量每个元素是优化变量的解
# outputFileName是写文件的路径
import numpy as np


def printSolutions(solutions, outputFileName):
    def toString(vector):
        result = ""
        for i in range(len(vector)):
            result = result + str(vector[i]) + "  "
        return result

    ofile = open(outputFileName, "w")
    for i in range(len(solutions)):
        ofile.write(toString(solutions[i]) + "\n")
    ofile.close()


def readMatrix(inputFileName):
    import os
    assert (os.path.exists(inputFileName))
    ifile = open(inputFileName, "r")
    string = ifile.readline()
    a = string.split()
    row = int(a[0])
    col = int(a[1])
    A = np.zeros((row, col))
    for i in range(row):
        string = ifile.readline()
        a = string.split()
        for j in range(len(a)):
            A[i, j] = float(a[j])
    ifile.close()
    return A


def readVector(inputFileName):
    import os
    assert (os.path.exists(inputFileName))
    ifile = open(inputFileName, "r")
    line = ifile.readline().strip()
    length = int(line)
    vector = np.zeros(length)
    for i in range(length):
        line = ifile.readline().strip()
        vector[i] = float(line)
    ifile.close()
    return vector


def printMatrix(matrix):
    (row, col) = matrix.shape
    for i in range(row):
        for j in range(col):
            print(matrix[i, j], "  ", )
        print("\n", )


def read(AFile, bFile, cFile, xFile):
    import os

    assert (os.path.exists(AFile))
    assert (os.path.exists(bFile))
    assert (os.path.exists(cFile))
    assert (os.path.exists(xFile))

    A = readMatrix(AFile)
    b = readVector(bFile)
    c = readVector(cFile)
    x = readVector(xFile)
    return A, b, c, x


# 穷举遍历所有可能的协作簇,输入终端设备的数量和匹配对数量，然后获得所有可能的协作簇结果
def permute(N, pairs_number):
    nums = [i for i in range(1, N + 1)]

    def backtrack(first):
        if first == n:
            res.append(nums[:])
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first + 1)
            nums[first], nums[i] = nums[i], nums[first]

    n = len(nums)
    res = []
    backtrack(0)
    pairs_array = []
    for k in range(len(res)):
        pairs_array_tmp = [[res[k][i], res[k][i + 1]] for i in range(0, pairs_number * 2, 2)]
        pairs_array.append(pairs_array_tmp)
    new_lst = [list(x) for x in set(tuple(tuple(y) for y in x) for x in pairs_array)]
    new_lst = [[list(x) for x in tup] for tup in new_lst]
    return res, new_lst


# 把二维向量转换为集合
def ToSet(lst):
    num_set = set()
    for sublist in lst:
        for num in sublist:
            num_set.add(num)
    return num_set


def main():
    A, b, c, x = read("data/A.txt", "data/b.txt", "data/c.txt", "data/x.txt")
    if True:
        print("c = ", c)
        print("A = ", A)
        print("b = ", b)
        print("x = ", x)


if __name__ == "__main__":
    import sys

    sys.exit(main())
