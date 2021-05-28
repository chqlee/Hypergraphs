import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy as sci

def initMatrix():
    matrix = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    return matrix

# draw the contagion network
# 可视化传播图
# @param G:  网络
# @param N:  最终总共有多少个节点
# @param s_list:    易感节点列表
# @param i_list:    感染节点列表
def drawGraph(G,N,s_list,i_list):

    options = {"node_size": 50, "alpha": 0.8}
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, nodelist=s_list, node_color="b", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=i_list, node_color="r", **options)
    nx.draw_networkx_edges(G, pos, alpha=1, width=1)
    plt.axis("off")
    plt.show()

def SI_Contagions(df_matrix_init, beta, inode, tmax):
    A = pd.DataFrame(np.dot(df_matrix_init, df_matrix_init.T))
    nodes_num = len(A)
    print(A)
    np.fill_diagonal(A.values, 0)
    print(A)

    G = nx.from_numpy_matrix(A.values)

    # np_nodes = np.sort(np.array(G.nodes))
    # inode_list = random.sample(list(np_nodes), 1)
    # inode = 2

    s_list = []
    i_list = []

    for each in G.nodes():
        if each != inode:
            s_list.append(each)
        else:
            i_list.append(each)


    df_adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).todense())
    print(df_adj_matrix)
    t_total_list = [0]
    i_total_list = [1]
    s_total_list = [nodes_num - 1]
    t = 1
    while t < tmax:
        for each in i_list:
            # print(each)
            # print(df_matrix_init.values[each])
            arr = np.array(df_matrix_init.values[each])
            arr_1 = np.where(arr == 1)[0]
            # print(np.where(arr == 1)[0])
            for index in arr_1:
                # print(index)
                # print(df_matrix_init[index])
                index_arr = np.array(df_matrix_init[index])
                # print(np.where(index_arr == 1)[0])
                arr_new = np.where(index_arr == 1)[0]
                for i in arr_new:
                    if i in s_list:
                        if random.random() < beta:
                            i_list.append(i)
                            s_list.remove(i)

        i_num = len(i_list)
        s_num = len(s_list)
        i_total_list.append(i_num)
        s_total_list.append(s_num)
        t_total_list.append(t)
        t = t + 1
    # drawGraph(G, nodes_num, s_list, i_list)
    # draw the graph
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=nodes_num)
    # plt.show()
    return (t_total_list, i_total_list, s_total_list)

def SIS_Contagions(df_matrix_init, beta, gamma, inode, tmax):
    A = pd.DataFrame(np.dot(df_matrix_init, df_matrix_init.T))
    nodes_num = len(A)
    print(A)
    np.fill_diagonal(A.values, 0)
    print(A)

    G = nx.from_numpy_matrix(A.values)

    # np_nodes = np.sort(np.array(G.nodes))
    # inode_list = random.sample(list(np_nodes), 1)
    # inode = 2

    s_list = []
    i_list = []

    for each in G.nodes():
        if each != inode:
            s_list.append(each)
        else:
            i_list.append(each)


    df_adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).todense())
    print(df_adj_matrix)
    t_total_list = [0]
    i_total_list = [1]
    s_total_list = [nodes_num - 1]
    t = 1
    while t < tmax:
        for each in i_list:
            # print(each)
            # print(df_matrix_init.values[each])
            arr = np.array(df_matrix_init.values[each])
            arr_1 = np.where(arr == 1)[0]
            # print(np.where(arr == 1)[0])
            for index in arr_1:
                # print(index)
                # print(df_matrix_init[index])
                index_arr = np.array(df_matrix_init[index])
                # print(np.where(index_arr == 1)[0])
                arr_new = np.where(index_arr == 1)[0]
                for i in arr_new:
                    if i in s_list:
                        if random.random() < beta:
                            i_list.append(i)
                            s_list.remove(i)
        for i in i_list:
            if random.random() < gamma:
                i_list.remove(i)
                s_list.append(i)
        i_num = len(i_list)
        s_num = len(s_list)
        i_total_list.append(i_num)
        s_total_list.append(s_num)
        t_total_list.append(t)
        t = t + 1
    # drawGraph(G, nodes_num, s_list, i_list)
    # draw the graph
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=nodes_num)
    # plt.show()
    return (t_total_list, i_total_list, s_total_list)

def SIR_Contagions(df_matrix_init, beta, gamma, inode, tmax):
    A = pd.DataFrame(np.dot(df_matrix_init, df_matrix_init.T))
    nodes_num = len(A)
    print(A)
    np.fill_diagonal(A.values, 0)
    print(A)

    G = nx.from_numpy_matrix(A.values)

    # np_nodes = np.sort(np.array(G.nodes))
    # inode_list = random.sample(list(np_nodes), 1)
    # inode = 2

    s_list = []
    i_list = []
    r_list = []

    for each in G.nodes():
        if each != inode:
            s_list.append(each)
        else:
            i_list.append(each)


    df_adj_matrix = pd.DataFrame(nx.adjacency_matrix(G).todense())
    print(df_adj_matrix)
    t_total_list = [0]
    i_total_list = [1]
    r_total_list = [0]
    s_total_list = [nodes_num - 1]
    t = 1
    while t < tmax:
        for each in i_list:
            # print(each)
            # print(df_matrix_init.values[each])
            arr = np.array(df_matrix_init.values[each])
            arr_1 = np.where(arr == 1)[0]
            # print(np.where(arr == 1)[0])
            for index in arr_1:
                # print(index)
                # print(df_matrix_init[index])
                index_arr = np.array(df_matrix_init[index])
                # print(np.where(index_arr == 1)[0])
                arr_new = np.where(index_arr == 1)[0]
                for i in arr_new:
                    if i in s_list:
                        if random.random() < beta:
                            i_list.append(i)
                            s_list.remove(i)
        for i in i_list:
            if random.random() < gamma:
                i_list.remove(i)
                r_list.append(i)
        r_num = len(r_list)
        i_num = len(i_list)
        s_num = len(s_list)
        r_total_list.append(r_num)
        i_total_list.append(i_num)
        s_total_list.append(s_num)
        t_total_list.append(t)
        t = t + 1
    # drawGraph(G, nodes_num, s_list, i_list)
    # draw the graph
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=False, node_size=nodes_num)
    # plt.show()
    return (t_total_list, i_total_list, s_total_list, r_total_list)

def SI(t):
    matrix_init = initMatrix()
    df_matrix_init = pd.DataFrame(matrix_init)
    print(df_matrix_init)
    beta = 0.016
    # SI_Contagions(df_matrix_init, beta, 5)
    N1 = 100

    s_sum = np.zeros(t)
    i_sum = np.zeros(t)
    s_matrix = []
    i_matrix = []
    # all nodes SI
    for inode in range(0, N1):
        res = SI_Contagions(df_matrix_init, beta, inode, t)
        d1 = res[1]
        d2 = res[2]
        s_matrix.append(d2)
        i_matrix.append(d1)
        i_sum = i_sum + d1
        s_sum = s_sum + d2

    # get the matrix to compute the error bar and the avg val
    df_s_matrix = pd.DataFrame(s_matrix)
    df_i_matrix = pd.DataFrame(i_matrix)
    s_std_list = []
    i_std_list = []

    for i in range(0, t):
        s_std_list.append(np.std(df_s_matrix[i]))
        i_std_list.append(np.std(df_i_matrix[i]))

    t_list = np.arange(t)
    s_list = s_sum / N1
    i_list = i_sum / N1

    std1 = np.mean(s_std_list)
    std2 = np.mean(i_std_list)

    x = t_list
    y = i_list
    y2 = s_list

    print(x)
    print(y)
    print(y2)
    # draw picture

    plt.xlim(0, t-1)
    plt.ylim(0, N1)
    plt.xlabel('t')
    plt.title('SI on Hypergraghs')
    plt.plot(x, y, color='r', label='i(t)')
    # plt.errorbar(x, y2, yerr=std2, ecolor='g')
    plt.plot(x, y2, color='b', label='s(t)')
    # plt.errorbar(x, y, yerr=std1, ecolor='y')

    i_error_1 = i_list - std2
    i_error_2 = i_list + std2
    plt.fill_between(x, i_error_1, i_error_2, color='r', alpha=0.2)
    s_error_1 = s_list - std1
    s_error_2 = s_list + std1
    plt.fill_between(x, s_error_1, s_error_2, color='b', alpha=0.2)

    plt.legend()
    plt.show()

def SIS(t):
    matrix_init = initMatrix()
    df_matrix_init = pd.DataFrame(matrix_init)
    print(df_matrix_init)
    beta = 0.03
    gamma = 0.2
    # SI_Contagions(df_matrix_init, beta, 5)
    N1 = 100

    s_sum = np.zeros(t)
    i_sum = np.zeros(t)
    s_matrix = []
    i_matrix = []
    # all nodes SI
    for inode in range(0, N1):
        res = SIS_Contagions(df_matrix_init, beta, gamma, inode, t)
        d1 = res[1]
        d2 = res[2]
        s_matrix.append(d2)
        i_matrix.append(d1)
        i_sum = i_sum + d1
        s_sum = s_sum + d2

    # get the matrix to compute the error bar and the avg val
    df_s_matrix = pd.DataFrame(s_matrix)
    df_i_matrix = pd.DataFrame(i_matrix)
    s_std_list = []
    i_std_list = []

    for i in range(0, t):
        s_std_list.append(np.std(df_s_matrix[i]))
        i_std_list.append(np.std(df_i_matrix[i]))

    t_list = np.arange(t)
    s_list = s_sum / N1
    i_list = i_sum / N1

    std1 = np.mean(s_std_list)
    std2 = np.mean(i_std_list)

    x = t_list
    y = i_list
    y2 = s_list

    print(x)
    print(y)
    print(y2)
    # draw picture

    plt.xlim(0, t-1)
    plt.ylim(0, N1)
    plt.xlabel('t')
    plt.title('SIS on Hypergraghs')
    plt.plot(x, y, color='r', label='i(t)')
    # plt.errorbar(x, y2, yerr=std2, ecolor='g')
    plt.plot(x, y2, color='b', label='s(t)')
    # plt.errorbar(x, y, yerr=std1, ecolor='y')

    i_error_1 = i_list - std2
    i_error_2 = i_list + std2
    plt.fill_between(x, i_error_1, i_error_2, color='r', alpha=0.2)
    s_error_1 = s_list - std1
    s_error_2 = s_list + std1
    plt.fill_between(x, s_error_1, s_error_2, color='b', alpha=0.2)

    plt.legend()
    plt.show()

def SIR(t):
    matrix_init = initMatrix()
    df_matrix_init = pd.DataFrame(matrix_init)
    print(df_matrix_init)
    beta = 0.02
    gamma = 0.15
    # SI_Contagions(df_matrix_init, beta, 5)
    N1 = 100

    s_sum = np.zeros(t)
    i_sum = np.zeros(t)
    r_sum = np.zeros(t)
    s_matrix = []
    i_matrix = []
    r_matrix = []
    # all nodes SI
    for inode in range(0, N1):
        res = SIR_Contagions(df_matrix_init, beta, gamma, inode, t)
        d1 = res[1]
        d2 = res[2]
        d3 = res[3]
        s_matrix.append(d2)
        i_matrix.append(d1)
        r_matrix.append(d3)
        i_sum = i_sum + d1
        s_sum = s_sum + d2
        r_sum = r_sum + d3

    # get the matrix to compute the error bar and the avg val
    df_s_matrix = pd.DataFrame(s_matrix)
    df_i_matrix = pd.DataFrame(i_matrix)
    df_r_matrix = pd.DataFrame(r_matrix)
    s_std_list = []
    i_std_list = []
    r_std_list = []

    for i in range(0, t):
        s_std_list.append(np.std(df_s_matrix[i]))
        i_std_list.append(np.std(df_i_matrix[i]))
        r_std_list.append(np.std(df_r_matrix[i]))

    t_list = np.arange(t)
    s_list = s_sum / N1
    i_list = i_sum / N1
    r_list = r_sum / N1

    std1 = np.mean(s_std_list)
    std2 = np.mean(i_std_list)
    std3 = np.mean(r_std_list)

    x = t_list
    y = i_list
    y2 = s_list
    y3 = r_list

    print(x)
    print(y)
    print(y2)
    print(y3)
    # draw picture

    plt.xlim(0, t-1)
    plt.ylim(0, N1)
    plt.xlabel('t')
    plt.title('SIR on Hypergraghs')
    plt.plot(x, y, color='r', label='i(t)')
    # plt.errorbar(x, y2, yerr=std2, ecolor='g')
    plt.plot(x, y2, color='b', label='s(t)')
    # plt.errorbar(x, y, yerr=std1, ecolor='y')
    plt.plot(x, y3, color='g', label='r(t)')

    i_error_1 = i_list - std2
    i_error_2 = i_list + std2
    plt.fill_between(x, i_error_1, i_error_2, color='r', alpha=0.2)
    s_error_1 = s_list - std1
    s_error_2 = s_list + std1
    plt.fill_between(x, s_error_1, s_error_2, color='b', alpha=0.2)
    r_error_1 = r_list - std3
    r_error_2 = r_list + std3
    plt.fill_between(x, r_error_1, r_error_2, color='g', alpha=0.2)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # matrix_init = initMatrix()
    # df_matrix_init = pd.DataFrame(matrix_init)
    # print(df_matrix_init)
    # beta = 0.1
    # SI_Contagions(df_matrix_init, beta, 5)

    # SI(16)
    SIS(16)
    # SIR(16)