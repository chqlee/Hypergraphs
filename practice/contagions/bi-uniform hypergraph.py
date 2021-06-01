import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def initMatrix(N, H, W):
    np_nodes_list = np.arange(500)
    nodes_list = list(np_nodes_list)

    matrix = np.zeros((N, int(N / H) + int(N / W)))
    df_matrix = pd.DataFrame(matrix)

    for i in range(0, int(N / H)):
        sample = random.sample(list(np.sort(nodes_list)), H)
        for each in sample:
            df_matrix[i][each] = 1
            nodes_list.remove(each)

    nodes_list = list(np_nodes_list)

    for i in range(int(N / H), int(N / H) + int(N / W)):
        sample = random.sample(list(np.sort(nodes_list)), W)
        for each in sample:
            df_matrix[i][each] = 1
            nodes_list.remove(each)
    return df_matrix

def fx(x,c):
    if(x >= 0 and x <= c):
        return x
    elif(x > c):
        return c

def SIS_bi_uniform_HyperG(J,N,M,inode,c):
    r = np.random.rand(1, N)
    r_list = r[0]
    x = np.zeros(N)
    x[inode] = 1
    tao = 0.18
    gamma = 1
    t = 1
    i_total_num = [1]
    t_total_num = [0]
    tmax = 15
    while t < tmax:
        for i in range(0, N):
            if x[i] == 0:
                sum = 0
                matrix_xj = np.dot(x, J)
                matrix_j = J.values
                for j in range(0, M):
                    fj = fx(matrix_xj[j], c)
                    sum = sum + (matrix_j[i][j]*fj)
                exponential = -1 * tao * sum
                res = 1 - math.exp(exponential)
                if random.random() < res:
                    x[i] = 1
        # for k in range(0, N):
            elif x[i] == 1:
                exponential = -1 * gamma
                res = 1 - math.exp(exponential)
                if random.random() < res:
                    x[i] = 0
        t = t + 1
        t_total_num.append(t)
        i_total_num.append(np.sum(np.array(x)))
    return t_total_num, i_total_num
    # x = t_total_num
    # y = i_total_num
    # print(x)
    # print(y)
    # plt.plot(x, y)
    # plt.show()

def compute(J, N, M, c):
    i_matrix = []
    t_matrix = []
    # for inode in range(0, 30):
    for inode in range(460, 500):
        t_total_num, i_total_num = SIS_bi_uniform_HyperG(J, N, M, inode, c)
        t_matrix.append(t_total_num)
        i_matrix.append(i_total_num)
        print('-------')
    df_i_matrix = pd.DataFrame(i_matrix)
    return np.array(df_i_matrix.loc[:N].mean())

if __name__ == '__main__':

    N = 500
    H = 5
    W = 10
    M = int(N / H) + int(N / W)
    J = initMatrix(N, H, W)
    t_matrix = []
    i_matrix = []
    c1 = 2
    c2 = 10
    c3 = 3
    y1 = compute(J, N, M, c1)
    y2 = compute(J, N, M, c2)
    y3 = compute(J, N, M, c3)
    x = np.arange(15)
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.xlabel('t')
    plt.ylabel('I(t)')
    plt.xlim(0, 15)
    plt.ylim(0, 300)
    plt.plot(x, y1, color='r', label='c=2', linestyle=':')
    plt.plot(x, y2, color='g', label='c=10', linestyle='-.')
    plt.plot(x, y3, color='b', label='c=3')
    plt.legend()
    plt.show()
