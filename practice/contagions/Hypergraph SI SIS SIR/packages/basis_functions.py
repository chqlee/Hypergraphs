import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class BasicFuncs:
    """
    基础方法封装类：用于基础方法函数等封装
    """

    def constructMatrix(self):
        """
        构造超图的点边矩阵
        :return: 超图的点边矩阵 matrix
        """
        matrix = np.random.randint(0, 2, size=(100, 10))
        for i in range(100):
            if sum(matrix[i]) == 0:
                j = np.random.randint(0, 10)
                matrix[i, j] = 1
        hyper_matrix = pd.DataFrame(matrix)
        N = len(hyper_matrix.index.values)
        return hyper_matrix, N

    def drawPic(self, total_matrix, total_matrix_R, type, strategy, theta, N):
        if type == 'SIR':
            final_I_list = pd.DataFrame(total_matrix).mean(axis=0) / N
            final_R_list = pd.DataFrame(total_matrix_R).mean(axis=0) / N
            final_S_list = 1 - final_I_list - final_R_list
        else:
            final_I_list = pd.DataFrame(total_matrix).mean(axis=0) / N
            final_S_list = 1 - final_I_list
        T_list = np.arange(len(final_I_list))
        if type == 'SI':
            # 计算均值并绘图
            plt.title("Hypergraph SI of " + strategy + " strategy     " + "beta:" + str(theta[0]))

        elif type == 'SIS':
            # 计算均值并绘图
            plt.title("Hypergraph SIS of " + strategy + " strategy     " + "beta:" + str(theta[0]) + "  gamma:" + str(theta[1]))

        elif type == 'SIR':
            # 计算均值并绘图
            plt.title("Hypergraph SIR of " + strategy + " strategy     " + "beta:" + str(theta[0]) + "  gamma:" + str(theta[1]))
            plt.plot(T_list, final_R_list, label='r(t)', color='g')

        plt.plot(T_list, final_I_list, label='i(t)', color='r')
        plt.plot(T_list, final_S_list, label='s(t)')
        plt.legend()
        plt.show()

