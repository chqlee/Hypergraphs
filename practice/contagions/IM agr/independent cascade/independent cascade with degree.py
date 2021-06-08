import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy

def loadData():
    """
    用于处理数据集
    :return:data
    """
    data_SYTH = pd.read_csv('dataset/sythetic.txt', sep=' ', index_col=False, header=None)
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

def getDegreeList(G,key_nodes_total):
    print(G.degree())
    degree_list = []
    for (i, j) in G.degree():
        degree_list.append(j)
    ser1 = pd.Series(degree_list).rank(method='first', ascending=False)
    degree_max_list = ser1[ser1.values <= key_nodes_total].index
    return degree_max_list

# draw the contagion network
# 可视化传播图
# @param G:  网络
# @param N:  最终总共有多少个节点
# @param s_list:    易感节点列表
# @param i_list:    感染节点列表
def drawGraph(G,N,s_list,i_list):

    options = {"node_size": 50, "alpha": 1}
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=s_list, node_color="b", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=i_list, node_color="r", **options)
    nx.draw_networkx_edges(G, pos, alpha=1, width=1)
    plt.axis("off")
    plt.show()

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



if __name__ == '__main__':

    # test the NetHEPT dataset
    N = 15223
    tmax = 10
    data_NetHEPT_df = loadData2('NetHEPT')
    DG = initGraph2(data_NetHEPT_df, N)
    seed_size = 50
    seed_size_list = np.arange(seed_size)
    inf_spread_matrix = []

    R = 200
    for r in range(0, R):
        inf_spread_list = []
        for i in range(1, seed_size + 1):
            DG = copy.deepcopy(DG)
            seed_list = getDegreeList(DG, i)
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
    plt.plot(x, y, marker='s', markersize=4., label='Degree', color='coral')
    plt.legend()
    plt.show()
