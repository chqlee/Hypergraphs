import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def getERNetwork(N,p):
    G = nx.random_graphs.erdos_renyi_graph(N, p)
    return G

# get the Nearest-neighbor coupled network
# 得到完全规则网络  最近邻耦合网络
# @param G:  图
# @param nodes_num:  网络的节点数
# @param k : 左右分别相连的邻居数
def initTheGraph(G, nodes_num, k):
    half_K = int(k / 2 + 1)
    for j in range(1, half_K):
        # link edges
        for i in range(0, nodes_num):
            from_node = i
            to_node = (i+j) % nodes_num
            G.add_edge(from_node, to_node)

    # draw the graph
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=nodes_num)
    # plt.show()

    return G

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

def startContagionsByDegree(G,seed_list,key_nodes_total,N):
    s_list = []
    i_list = []


    t_total_list = [0]
    i_total_list = [key_nodes_total]
    s_total_list = [N - key_nodes_total]

    # 先将度最大的几个节点设置成种子节点
    for each in range(0, N):
        if each in seed_list:
            i_list.append(each)
        else:
            s_list.append(each)

    print(i_list)
    t = 1
    tmax = 5
    beta = 0.2
    drawGraph(G, N, s_list, i_list)
    while t <= tmax:
        for each in i_list:
            for adj in G[each]:
                if adj in s_list:
                    if random.random() < beta:
                        s_list.remove(adj)
                        i_list.append(adj)
        i_num = len(i_list)
        s_num = len(s_list)
        print(i_num)
        drawGraph(G, N, s_list, i_list)
        i_total_list.append(i_num)
        s_total_list.append(s_num)
        t_total_list.append(t)
        t = t + 1
    print(i_total_list)


def startContagionsByGreedy(G, key_nodes_total, N, tmax):
    """
    基于SI传播的Greedy算法
    :param G:待传播的网络
    :param key_nodes_total:总共要找的seed节点数
    :param N:网络节点总数
    :param tmax:时间长度
    :return:
    """
    key_list = []
    remain_nodes = [i for i in range(N)]

    beta = 0.2

    for k in range(0, key_nodes_total):
        global maxNodeIndex
        maxNodeNum = 0
        for each in key_list:
            if each in remain_nodes:
                remain_nodes.remove(each)
        for inode in remain_nodes:
            s_list = []
            i_list = []

            if len(key_list) > 0:
                for each in G.nodes():
                    if each in key_list:
                        i_list.append(each)
                    else:
                        s_list.append(each)

            for each in G.nodes():
                if each != inode:
                    s_list.append(each)
                else:
                    i_list.append(each)
            t = 1
            while t < tmax:
                for each in i_list:
                    for adj in G[each]:
                        if adj in s_list:
                            if random.random() < beta:
                                s_list.remove(adj)
                                i_list.append(adj)
                t = t + 1
            # 每次结束后i的总数
            i_num = len(i_list)
            # 找到每次传播最大的节点 赋值
            if maxNodeNum < i_num:
                maxNodeIndex = inode
                maxNodeNum = i_num
        # 将影响传播最大的节点加入
        key_list.append(maxNodeIndex)
        print('key_list', key_list)



if __name__ == '__main__':
    N = 20
    p = 0.1
    G = getERNetwork(N, p)
    # G = initTheGraph(nx.Graph(), 20, 4)
    key_nodes_total = 3
    seed_list = getDegreeList(G, key_nodes_total)
    # startContagionsByDegree(G, seed_list, key_nodes_total, N)
    tmax = 5
    startContagionsByGreedy(G, key_nodes_total, N, tmax)