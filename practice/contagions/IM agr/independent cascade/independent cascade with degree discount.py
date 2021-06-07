import networkx as nx
import numpy as np
import pandas as pd
import random
import copy
import matplotlib.pyplot as plt

def loadData():
    """
    用于处理数据集
    :return:data
    """
    data_SYTH = pd.read_csv('./dataset/sythetic.txt', sep=' ', index_col=False, header=None)
    data = pd.DataFrame(data_SYTH)
    return data

def loadData2(data):
    """
    该方法用于 读入处理数据集的数据
    :param data: 读取哪一个数据
    :return: dataset
    """
    global dataset
    if data == 'NetHEPT':
        data_NetHEPT = pd.read_csv('dataset/hep.txt', sep=' ', index_col=False, header=None)
        data_NetHEPT_df = pd.DataFrame(data_NetHEPT)
        dataset = data_NetHEPT_df
    elif data == 'NetPHY':
        data_NetPHY = pd.read_csv('dataset/phy.txt', sep=' ', index_col=False, header=None)
        data_NetPHY_df = pd.DataFrame(data_NetPHY)
        dataset = data_NetPHY_df
    return dataset

def initGraph(data_df,nodes_num):
    """
    该方法用于 根据数据集 生成 有向图
    :param data_df:处理好的数据集
    :param nodes_num:网络总节点数
    :return:DG
    """
    DG = nx.DiGraph()
    for each in range(0, nodes_num):
        DG.add_node(each)
    for row in data_df.values:
        DG.add_weighted_edges_from([(int(row[0]-1), int(row[1]-1), row[2])])
    return DG

def initGraph2(data_NetHEPT_df,nodes_num):
    """
    该方法用于 根据数据集 生成 有向图
    :param data_NetHEPT_df:处理好的数据集
    :param nodes_num:网络总节点数
    :return:DG
    """
    DG = nx.DiGraph()
    for each in range(0, nodes_num):
        DG.add_node(each)
    for row in data_NetHEPT_df.values:
        DG.add_weighted_edges_from([(int(row[0]-1), int(row[1]-1), 0.01)])
    return DG

def IC_Contagions(DG,seed_list,N,tmax):
    """
    独立级联 传播方式 实现算法
    :param DG:有向网络
    :param seed_list:种子节点列表
    :param N:节点总数
    :return:list
    """
    s_list = []
    i_list = []
    for each in DG.nodes:
        if each in seed_list:
            i_list.append(each)
        else:
            s_list.append(each)

    t_total_list = [0]
    i_total_list = [len(seed_list)]
    s_total_list = [N - len(seed_list)]

    t = 1

    while t < tmax:
        for each in i_list:
            for adj in list(DG.neighbors(each)):
                if adj in s_list:
                    if random.random() < DG.get_edge_data(each, adj)['weight']:
                        s_list.remove(adj)
                        i_list.append(adj)
        i_num = len(i_list)
        s_num = len(s_list)
        i_total_list.append(i_num)
        s_total_list.append(s_num)
        t_total_list.append(t)
        t = t + 1
        print('------computing------')

    return (t_total_list, i_total_list, s_total_list)

def degreeDiscountIC(DG, k):
    """
    该方法为 基于IC传播 的 Degree Discount算法
    :param DG: 待传播有向网络
    :param k: 需要寻找的种子节点的个数
    :return: key_list
    """
    key_list = []
    d_list = []
    dd_list = []
    t_list = []
    for each in DG.nodes:
        d_list.append(DG.degree(each))
        dd_list.append(DG.degree(each))
        t_list.append(0)
    for i in range(0, k):
        remain_nodes = []
        dd_list_new = copy.deepcopy(dd_list)
        for inode in DG.nodes:
            if inode not in key_list:
                remain_nodes.append(inode)
            else:
                dd_list_new[inode] = 0
        u = dd_list_new.index(max(dd_list_new))
        key_list.append(u)
        remain_nodes.remove(u)
        for adj in list(DG.neighbors(u)):
            if adj in remain_nodes:
                t_list[adj] = t_list[adj] + 1
                val = DG.get_edge_data(u, adj)['weight']
                dd_list[adj] = d_list[adj] - (2 * t_list[adj]) - ((d_list[adj] - t_list[adj]) * t_list[adj] * val)
    print('t_list', t_list)
    print('d_list', d_list)
    print('dd_list', dd_list)
    return key_list

if __name__ == '__main__':
    # test the NetHEPT dataset
    N = 15223
    tmax = 10
    data_NetHEPT_df = loadData2('NetHEPT')
    DG = initGraph2(data_NetHEPT_df, N)
    # seed_list = random.sample(list(np.arange(N)), 500)


    # N = 9
    # tmax = 5
    # data_SYTH_df = loadData()
    # DG = initGraph(data_SYTH_df, N)
    # key_nodes_total = 3

    # print('------------------------------------------')
    # print('经 Degree Discount 算法，您需寻找的' + str(key_nodes_total) + '个种子节点如下')
    # print(seed_list)
    seed_size = 50
    seed_size_list = np.arange(seed_size)
    inf_spread_matrix = []
    R = 100
    for r in range(0, R):
        inf_spread_list = []
        for i in range(1, seed_size+1):
            DG = copy.deepcopy(DG)
            seed_list = degreeDiscountIC(DG, i)
            print('-------------------- ROUND ' + str(r) + ' ----------------------')
            print('经 Degree Discount 算法，您需寻找的' + str(i) + '个种子节点如下')
            print(seed_list)
            t_total_list, i_total_list, s_total_list = IC_Contagions(DG, seed_list, N, tmax)
            inf_spread_list.append(i_total_list[-1])
        inf_spread_matrix.append(inf_spread_list)
    mean_arr = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    print(mean_arr)
    x = seed_size_list
    y = mean_arr
    plt.ylabel('Influence spread')
    plt.xlabel('seed set size')
    plt.plot(x, y, marker='s', markersize=4., label='DegreeDiscountIC')
    plt.legend()
    plt.show()

