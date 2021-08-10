import numpy as np
import random

class ProcessFuncs:
    """
    策略过程封装类：用于不同策略寻找可能感染的节点过程及筛选过程
    """

    def findAdjNode_RP(self, I_list, df_hyper_matrix):
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
        nodes_in_edges = np.array([])
        for edge in edges_conclude_nodes:
            nodes = np.where(np.array(df_hyper_matrix[edge]) == 1)[0]
            nodes_in_edges = np.append(nodes_in_edges, nodes)
        return np.unique(nodes_in_edges)


    def findAdjNode_CP(self, I_list, df_hyper_matrix):
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
        return nodes


    def formatInfectedList(self, I_list, infected_list):
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


    def getTrueStateNode(self, adj_nodes, I_list, R_list):
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