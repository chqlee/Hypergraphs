from basis_functions import BasicFuncs
from spreading_functions import SpreadFuncs
import time

class Main:
    def startMain(theta, iters, strategy, type):
        """
        :param theta: 参数列表
        :param iters: 迭代次数
        :param strategy: 策略
        :param type: 传播类型
        :return: void
        """
        start = time.perf_counter()

        # 初始化
        bs = BasicFuncs()
        sp = SpreadFuncs()

        # 构造超图矩阵
        df_hyper_matrix, N = bs.constructMatrix()

        # 初始态赋值一个感染节点
        total_matrix, total_matrix_R = sp.spread(N, df_hyper_matrix, theta, iters, type, strategy)

        # 计算均值并绘图
        bs.drawPic(total_matrix, total_matrix_R, type, strategy, theta, N)

        end = time.perf_counter()
        print("算法时长：", str(end - start))
