# @Title    : 超图上的传播
# @Author   : tony
# @Date     : 2021/8/5
# @Dec      : CP strategy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

def constructMatrix():
    """
    构造超图的点边矩阵
    :return: 超图的点边矩阵 matrix
    """
    matrix = np.random.randint(0, 2, size=(100, 10))
    for i in range(100):
        if sum(matrix[i]) == 0:
            j = np.random.randint(0, 10)
            matrix[i, j] = 1
    return matrix

def findAdjNode(I_list, df_hyper_matrix):
    """
    找到邻居节点集合
    :param I_list: 感染节点集
    :param df_hyper_matrix: 超图的点边矩阵
    :return: 不重复的邻居节点集 np.unique(nodes_in_edges)
    """
    # 找到该点所属的超边集合
    edges_conclude_nodes = np.array([])
    for node in I_list:
        edges_conclude_nodes = np.where(np.array(df_hyper_matrix.loc[node]) == 1)[0]
    # 找到可能传播到超边中的顶点集合
    edge = random.sample(list(edges_conclude_nodes), 1)[0]
    nodes = np.where(np.array(df_hyper_matrix[edge]) == 1)[0]
    # print(nodes)
    return nodes

def formatInfectedList(I_list, infected_list):
    """
    筛选出不在I_list当中的节点
    :param I_list: 感染节点集
    :param infected_list: 本次受感染的节点（未筛选）
    :return: 本次受感染的节点（筛选后）format_list
    """
    format_list = []
    for i in range(0, len(infected_list)):
        if infected_list[i] not in I_list:
            format_list.append(infected_list[i])
    return format_list

def getTrueStateNode(adj_nodes, I_list, R_list):
    """
    从所有可能感染节点中排查筛选只是S态的节点
    :param adj_nodes: 所有可能感染节点
    :param I_list: 截至上一时刻全部感染节点
    :param R_list: 截至上一时刻全部恢复节点
    :return:
    """
    adj_list = list(adj_nodes)
    for i in range(0, len(adj_nodes)):
        if adj_nodes[i] in I_list or adj_nodes[i] in R_list:
            adj_list.remove(adj_nodes[i])
    return np.array(adj_list)

if __name__ == '__main__':
    start = time.perf_counter()

    # 构造超图矩阵
    hyper_matrix = constructMatrix()
    df_hyper_matrix = pd.DataFrame(hyper_matrix)

    # 初始态赋值一个感染节点
    N = len(df_hyper_matrix.index.values)
    total_matrix = []
    total_matrix_R = []
    for i_node in range(0, N):
        print("---------computing----------")
        # i_node = random.randint(0, N-1)
        I_list = [i_node]
        R_list = []

        # 开始传播
        beta = 0.2
        gamma = 0.1
        iters = 25
        I_total_list = [1]
        R_total_list = [0]

        for t in range(0, iters):
            # 找到邻居节点集
            adj_nodes = findAdjNode(I_list, df_hyper_matrix)
            # 排查筛选只是S态的节点
            adj_nodes = getTrueStateNode(adj_nodes, I_list, R_list)
            # 开始对邻节点传播
            random_list = np.random.random(size=len(adj_nodes))
            index_list = np.where(random_list < beta)[0]
            infected_list = adj_nodes[index_list]
            infected_list_unique = formatInfectedList(I_list, infected_list)
            # 上次感染的节点开始恢复
            for each in I_list:
                if random.random() < gamma and each not in R_list:
                    I_list.remove(each)
                    R_list.append(each)
            # 加入本次所感染的节点
            I_list.extend(infected_list_unique)
            I_total_list.append(len(I_list))
            R_total_list.append(len(R_list))
        total_matrix.append(I_total_list)
        total_matrix_R.append(R_total_list)

    # 计算均值并绘图
    final_I_list = pd.DataFrame(total_matrix).mean(axis=0) / N
    final_R_list = pd.DataFrame(total_matrix_R).mean(axis=0) / N
    final_S_list = 1 - final_I_list - final_R_list
    T_list = np.arange(len(final_I_list))
    plt.title("Hypergraph SIR of CP strategy     " + "beta:" + str(beta) + "  gamma:" + str(gamma))
    plt.plot(T_list, final_I_list, label='i(t)', color='r')
    plt.plot(T_list, final_R_list, label='r(t)', color='g')
    plt.plot(T_list, final_S_list, label='s(t)')
    plt.legend()
    plt.show()

    end = time.perf_counter()
    print(str(end - start))

