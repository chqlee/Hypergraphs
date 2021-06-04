import numpy as np
import pandas as pd
import random
import networkx as nx
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
        data_NetHEPT = pd.read_csv('./dataset/hep.txt', sep=' ', index_col=False, header=None)
        data_NetHEPT_df = pd.DataFrame(data_NetHEPT)
        dataset = data_NetHEPT_df
    elif data == 'NetPHY':
        data_NetPHY = pd.read_csv('./dataset/phy.txt', sep=' ', index_col=False, header=None)
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
    :param p:独立级联模型的 p值
    :return:DG
    """
    DG = nx.DiGraph()
    for each in range(0, nodes_num):
        DG.add_node(each)
    for row in data_NetHEPT_df.values:
        DG.add_weighted_edges_from([(int(row[0]-1), int(row[1]-1), 0.01)])
    return DG

def LT_Contagions(DG,seed_list,N,tmax):
    """
    线性阈值 传播方式 实现算法
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
                    if random.random() < DG.in_degree(2, weight='weight'):
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


def drawPicture(t_total_list, i_total_list, s_total_list, N):
    """
    该方法用于绘图
    :param t_total_list:
    :param i_total_list:
    :param s_total_list:
    :param N:
    :return:
    """
    x = t_total_list
    # y1 = i_total_list
    # y2 = s_total_list
    y1 = np.array(i_total_list) / N
    y2 = np.array(s_total_list) / N
    plt.title('MODEL: Linear Threshold')
    plt.xlabel('t')
    # plt.ylim(0, N)
    plt.ylim(0, 1)
    plt.plot(x, y1, color='r', label='i(t)')
    plt.plot(x, y2, color='b', label='s(t)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # test the NetHEPT dataset
    N = 15223
    tmax = 500
    data_NetHEPT_df = loadData2('NetHEPT')
    DG = initGraph2(data_NetHEPT_df, N)
    seed_list = random.sample(list(np.arange(N)), 1000)

    # N = 9
    # tmax = 5
    # data_SYTH_df = loadData()
    # DG = initGraph(data_SYTH_df, N)
    # seed_list = [0, 1]
    t_total_list, i_total_list, s_total_list = LT_Contagions(DG, seed_list, N, tmax)
    print(i_total_list)
    print(s_total_list)

    drawPicture(t_total_list, i_total_list, s_total_list, N)
