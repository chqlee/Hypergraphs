import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hyperspreading import Hyperspreading


def getSeedList_staticGD(degree, i):
    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    return df_sort_matrix.loc['node_index'][:i + 1]

def staticGreedy(df_hyper_matrix, K, R):
    # StaticGreedy：度贪婪 依次选择度最大的节点
    degree = df_hyper_matrix.sum(axis=1)
    inf_spread_matrix = []
    for r in range(0, R):
        scale_list = []
        for i in range(0, K):
            seeds = getSeedList_staticGD(degree, i)
            scale, I_list = hs.hyperSI(df_hyper_matrix, seeds)
            scale_list.append(scale)
            print("▋▋▋▋▋▋", end="")
        inf_spread_matrix.append(scale_list)
        print("\n")
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    print(final_scale_list)
    plt.plot(np.linspace(1, K, num=K), final_scale_list, marker='o', label='StaticGreedy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    hs = Hyperspreading()
    # 构造超图矩阵
    hyper_matrix = hs.constructMatrix()
    df_hyper_matrix = pd.DataFrame(hyper_matrix)

    K = 20
    iters = 100
    staticGreedy(df_hyper_matrix, K, iters)