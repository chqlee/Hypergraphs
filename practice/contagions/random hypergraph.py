import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

def initMatrix(N, M):
    print('init')
    node_list_np = np.arange(M)
    matrix = []
    node_list = list(np.sort(node_list_np))
    for i in range(0, N):
        zero_list = np.zeros(M)

        # print(zero_list)
        if i < 9:
            sample = random.sample((node_list), 8)
            for each in sample:
                zero_list[each] = 1
            matrix.append(zero_list)
        else:
            df_matrix = pd.DataFrame(matrix)
            print(df_matrix.sum(axis=0))
            sum_arr = np.array(df_matrix.sum(axis=0))
            index_arr = np.where(sum_arr >= 10)[0]
            print('-----', index_arr)
            for k in index_arr:
                if k in node_list:
                    node_list.remove(k)
            sample = random.sample((node_list), 8)
            print('&&&&&&&&&&', len(node_list))
            for each in sample:
                zero_list[each] = 1
            matrix.append(zero_list)
    df_matrix = pd.DataFrame(matrix)
    print(df_matrix.sum(axis=1))
    print(df_matrix.sum(axis=0))
    index_sum = np.array(df_matrix.sum(axis=1))
    col_sum = np.array(df_matrix.sum(axis=0))
    print(np.max(index_sum))
    print(np.min(index_sum))
    print(np.max(col_sum))
    print(np.min(col_sum))




if __name__ == '__main__':
    N = 500
    M = 400
    initMatrix(N, M)