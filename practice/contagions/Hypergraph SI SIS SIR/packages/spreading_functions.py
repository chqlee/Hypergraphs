from process_functions import ProcessFuncs
import numpy as np
import random

class SpreadFuncs:
    """
    传播过程封装类：用于超图上不同策略不同传播方式的模拟过程
    """
    def infect(self, df_hyper_matrix, I_list, R_list, theta, strategy):
        """
        开始对邻居可能被感染的节点传播
        :param df_hyper_matrix: 超图的点边矩阵
        :param I_list: 感染节点集
        :param R_list: 恢复节点集
        :param theta: 感染概率参数列表
        :param strategy: 策略
        :return: infected_list_unique
        """
        pf = ProcessFuncs()
        # 根据所给策略找到邻居节点集
        if strategy == 'RP':
            adj_nodes = pf.findAdjNode_RP(I_list, df_hyper_matrix)
        elif strategy == 'CP':
            adj_nodes = pf.findAdjNode_CP(I_list, df_hyper_matrix)
        # 排查筛选只是S态的节点
        adj_nodes = pf.getTrueStateNode(adj_nodes, I_list, R_list)
        # 开始对邻节点传播
        random_list = np.random.random(size=len(adj_nodes))
        index_list = np.where(random_list < theta[0])[0]
        infected_list = adj_nodes[index_list]
        infected_list_unique = pf.formatInfectedList(I_list, infected_list)
        return infected_list_unique

    def recover_SIR(self, I_list, R_list, theta):
        """
        SIR系列恢复
        :param I_list: 感染节点集
        :param R_list: 恢复节点集
        :param theta: 感染概率参数列表
        :return: void
        """
        # 上次感染的节点开始恢复
        for each in I_list:
            if random.random() < theta[1] and each not in R_list:
                I_list.remove(each)
                R_list.append(each)

    def rocover_SIS(self, I_list, theta):
        """
        SIS系列恢复
        :param I_list: 感染节点集
        :param theta: 感染概率参数列表
        :return: void
        """
        # 上次感染的节点开始恢复
        for each in I_list:
            if random.random() < theta[1]:
                I_list.remove(each)

    def spread(self, N, df_hyper_matrix, theta, iters, type, strategy):
        """
        传播主体部分
        """
        total_matrix = []
        total_matrix_R = []
        for i_node in range(0, N):
            print("▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋▋")
            I_list = [i_node]
            R_list = []
            I_total_list = [1]
            R_total_list = [0]

            for t in range(0, iters):
                # 传播
                infected_list_unique = SpreadFuncs.infect(self, df_hyper_matrix, I_list, R_list, theta, strategy)
                # 恢复
                if type == 'SIR':
                    SpreadFuncs.recover_SIR(self, I_list, R_list, theta)
                elif type == 'SIS':
                    SpreadFuncs.rocover_SIS(self, I_list, theta)
                # 加入本次所感染的节点
                I_list.extend(infected_list_unique)
                I_total_list.append(len(I_list))
                if type == 'SIR':
                    R_total_list.append(len(R_list))
            total_matrix.append(I_total_list)
            if type == 'SIR':
                total_matrix_R.append(R_total_list)
            print("-------------------------------------")
        return total_matrix, total_matrix_R