import networkx as nx
import numpy as np
import pandas as pd
import random
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

def drawGraph(G,s_list,i_list):
    """
    该方法用于可视化传播图
    :param G: 网络
    :param s_list: 易感节点列表
    :param i_list: 感染节点列表
    :return:
    """
    options = {"node_size": 50, "alpha": 1}
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=s_list, node_color="b", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=i_list, node_color="r", **options)
    nx.draw_networkx_edges(G, pos, alpha=1, width=1)
    plt.axis("off")
    plt.show()


def LT_Greedy(DG, key_nodes_total, N, tmax, R):
    """
    基于Linear Threshold传播方式的Greedy算法
    :param DG:待传播的有向网络
    :param key_nodes_total:总共要找的seed节点数
    :param N:网络节点总数
    :param tmax:时间长度
    :param R:节点循环次数
    :return:
    """
    key_list = []
    remain_nodes = [i for i in range(N)]

    for k in range(0, key_nodes_total):     # Greedy算法第一层（总共要找k个种子节点）
        global maxNodeIndex
        maxNodeNum = 0
        for each in key_list:
            if each in remain_nodes:
                remain_nodes.remove(each)
        for inode in remain_nodes:          # Greedy算法第二层（遍历每一个未被添加到中子节点群的节点）
            influent_sum = 0
            for i in range(0, R):           # Greedy算法第三层：（循环层，循环找出影响相对大的节点）
                s_list = []
                i_list = []

                if len(key_list) > 0:
                    for each in DG.nodes():
                        if each in key_list:
                            i_list.append(each)
                        else:
                            s_list.append(each)

                for each in DG.nodes():
                    if each != inode:
                        s_list.append(each)
                    else:
                        i_list.append(each)
                t = 1
                while t < tmax:
                    for each in i_list:
                        for adj in list(DG.neighbors(each)):
                            if adj in s_list:
                                if random.random() < DG.in_degree(2, weight='weight'):
                                    s_list.remove(adj)
                                    i_list.append(adj)
                    t = t + 1
                # 每次结束后i的总数
                i_num = len(i_list)
                influent_sum = influent_sum + i_num
                drawGraph(DG, s_list, i_list)
            # 找到每次传播平均值最大（提高准确性）的节点 赋值
            if maxNodeNum < influent_sum / R:
                maxNodeIndex = inode
                maxNodeNum = influent_sum / R
                print('maxNodeIndex', maxNodeIndex)
                print('maxNodeNum', maxNodeNum)
        # 将影响传播最大的节点加入
        key_list.append(maxNodeIndex)
        print('key_list', key_list)



if __name__ == '__main__':

    # init the val
    N = 9
    tmax = 5

    # load dataset
    data_SYTH_df = loadData()

    # init the network
    DG = initGraph(data_SYTH_df, N)

    # start greedy agr based on LT
    seed_list = [0, 1]
    key_nodes_total = 3
    R = 10
    LT_Greedy(DG, key_nodes_total, N, tmax, R)