import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Hyperspreading import Hyperspreading
import copy

def getSeedList_dynamicGD(degree, i, seed_list):
    matrix = []
    matrix.append(np.arange(len(degree)))
    matrix.append(degree)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.index = ['node_index', 'node_degree']
    df_sort_matrix = df_matrix.sort_values(by=df_matrix.index.tolist()[1], ascending=False, axis=1)
    global x
    for i in range(0, len(degree)):
        x = list(df_sort_matrix.loc['node_index'][i:i+1])[0]
        if x not in seed_list:
            return [x]
        else:
            # print(222)
            continue


def dynamicGreedy(df_hyper_matrix, K, R):
    # DynamicGreedy：节点的动态度贪婪 依次选择每次度最大的节点
    df_hyper_matrix_copy = copy.deepcopy(df_hyper_matrix)
    inf_spread_matrix = []

    for r in range(0, R):
        scale_list = []
        seed_list = []
        for i in range(0, K):
            # print(df_hyper_matrix)
            degree = df_hyper_matrix.sum(axis=1)
            seeds = getSeedList_dynamicGD(degree, i, seed_list)
            seed_list.append(seeds[0])
            # print(seed_list)
            scale, I_list = hs.hyperSI(df_hyper_matrix_copy, seeds)
            scale_list.append(scale)
            # 将seeds感染所涉及到的超边列全部置0
            edge_arr = np.where(np.array(df_hyper_matrix.loc[seeds[0]]))
            for eage in edge_arr:
                df_hyper_matrix[eage] = 0
            # print(I_list)
            # for inode in I_list:
            #     edge_arr = np.where(np.array(df_hyper_matrix.loc[inode]))
            #     for eage in edge_arr:
            #         df_hyper_matrix[eage] = 0
            print("▋▋▋▋▋▋", end="")
        inf_spread_matrix.append(scale_list)
        print("\n")
    final_scale_list = pd.DataFrame(inf_spread_matrix).mean(axis=0)
    print(final_scale_list)
    plt.plot(np.linspace(1, K, num=K), final_scale_list, color='r', marker='o', label='DynamicGreedy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    hs = Hyperspreading()
    # 构造超图矩阵
    hyper_matrix = hs.constructMatrix()
    df_hyper_matrix = pd.DataFrame(hyper_matrix)

    K = 20
    iters = 100
    dynamicGreedy(df_hyper_matrix, K, iters)