import networkx as nx
import numpy as np
import pandas as pd
import random
import copy

def loadData():
    """
    用于处理数据集
    :return:data
    """
    data_SYTH = pd.read_csv('./independent cascade/dataset/sythetic.txt', sep=' ', index_col=False, header=None)
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
        u = dd_list.index(max(dd_list_new))
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

    N = 9
    tmax = 5
    data_SYTH_df = loadData()
    DG = initGraph(data_SYTH_df, N)
    key_nodes_total = 3
    seed_list = degreeDiscountIC(DG, key_nodes_total)
    print('------------------------------------------')
    print('经 Degree Discount 算法，您需寻找的' + str(key_nodes_total) + '个种子节点如下')
    print(seed_list)
