import numpy as np
import pandas as pd
import random
import networkx as nx

def loadData(data):
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

def initGraph(data_NetHEPT_df,nodes_num,p):
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
        DG.add_weighted_edges_from([(row[0], row[1], p)])
    return DG

if __name__ == '__main__':

    # 初始化数据
    data_NetHEPT_df = loadData('NetHEPT')

    # 根据数据集构造特定网络
    DG = initGraph(data_NetHEPT_df, 15233, 0.01)