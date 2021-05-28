import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def initMatrix():
    """
    方法：初始化超图的点边邻接关系矩阵
    :return: matrix
    """
    matrix = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ]
    return matrix

def createHyperG(df_matrix_init):
    """
    方法：用于超图的创建（根据 matrix 点边邻接关系矩阵）和可视化
    :param df_matrix_init:
    :return: A
    """
    A = pd.DataFrame(np.dot(df_matrix_init, df_matrix_init.T))
    nodes_num = len(A)
    print(A)
    np.fill_diagonal(A.values, 0)
    print(A)
    G = nx.from_numpy_matrix(A.values)
    drawHyperG(G, nodes_num)
    return A

def drawHyperG(G, nodes_num):
    """
    方法：用于画超图的结构
    :param G: 超图的邻接矩阵表示
    :param nodes_num: 节点总数
    :return: void
    """
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=nodes_num)
    plt.show()


if __name__ == '__main__':
    # get the matrix about the nodes and edges
    matrix_init = initMatrix()
    df_matrix_init = pd.DataFrame(matrix_init)

    # create the HyperGraph
    createHyperG(df_matrix_init)
