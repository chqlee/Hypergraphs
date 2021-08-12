
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def GenerateHg(N, e):
    """
    随机生成一个N个节点e条超边的超图
    :param N: the number of nodes
    :param e: the number of hpe
    :return: the incident matrix of the hg
    """
    matrix = np.random.randint(0, 2, size=(N, e))
    for i in range(N):
        if sum(matrix[i]) == 0:
            j = np.random.randint(0, 10)
            matrix[i, j] = 1
    return matrix

def GetNodes(nodes, matrix):
    """
    选出一组节点的邻节点，CP机制
    """
    nbnodes = []
    for v in nodes:
        hpe = np.random.choice(np.where(matrix[v,:]==1)[0])
        nbnodes.extend(u for u in np.where(matrix[:,hpe] == 1)[0] if u not in nbnodes)


def GetHpe(v, matrix):
    """
    选出节点v所在的超边
    :param node: node v
    :param matrix: the incident matrix of the hg
    :return: the array of hpe that incident to node v
    """
    return np.where(matrix[v, :] == 1)[0]


def GetNodesofHpe(hpe, matrix):
    """
    选出一条超边所包含的节点
    :param hpe: a single hpe e
    :return: an array of nodes that are in the hpe e
    """
    return np.where(matrix[:, hpe] == 1)[0]


def GetNodesofHpes(hpes, matrix):
    """
    选出一组超边所包含的节点
    """
    nbnodes = []
    for hpe in hpes:
        nbnodes.extend(n for n in np.where(matrix[:, hpe] == 1)[0] if n not in nbnodes)
    return nbnodes

def ChooseHpe(hpes):
    """
    从若干条超边中随机选择一条
    """
    return np.random.choice(hpes)


def ContactNodes(nodes, beta):
    """
    给定点集，判断是否感染它们，并返回感染的点的索引
    """
    r = np.random.random(len(nodes))
    contacted_nodes = [nodes[i] for i in np.where(r < beta)[0]]
    return contacted_nodes


def RemoveNode(nodes, gamma):
    """
    给定点集，判断是否移除他们，并返回移除点的索引
    """
    r = np.random.random(len(nodes))
    removed_nodes = [nodes[i] for i in np.where(r < gamma)[0]]
    return removed_nodes


def SICP(N, e, time, beta):
    matrix = GenerateHg(N, e)
    ift_rates = pd.DataFrame(np.zeros((N, time)))
    ift_rates.iloc[:, 0] = 1 / N
    spt_rates = pd.DataFrame(np.zeros((N, time)))
    spt_rates.iloc[:, 0] = 1 - 1 / N

    for init_node in tqdm(range(100), desc="Loading..."):
        ift_nodes = [init_node]
        spt_nodes = [s for s in range(N) if s not in ift_nodes]
        for t in range(1, time):
            temp_i = []
            for i in ift_nodes:
                hpes = GetHpe(i, matrix)
                hpe = ChooseHpe(hpes)
                nbnodes = GetNodesofHpe(hpe, matrix)
                temp_i.extend(c for c in ContactNodes(nbnodes, beta) if c not in temp_i and c not in ift_nodes)
            ift_nodes.extend(i for i in temp_i)
            spt_nodes = [s for s in range(N) if s not in ift_nodes]
            ift_rates.iloc[init_node,t] = len(ift_nodes)/N
            spt_rates.iloc[init_node, t] = len(spt_nodes)/N

    return ift_rates.mean(axis=0), spt_rates.mean(axis=0)


def SIRP(N, e, time, beta):
    matrix = GenerateHg(N, e)
    ift_rates = pd.DataFrame(np.zeros(shape=(N,time)))
    ift_rates.iloc[:, 0] = 1 / N
    spt_rates = pd.DataFrame(np.zeros(shape=(N, time)))
    spt_rates.iloc[:, 0] = 1 - 1 / N

    for init_node in tqdm(range(100), desc="Loading..."):
        ift_nodes = [init_node]
        spt_nodes = [s for s in range(N) if s not in ift_nodes]
        for t in range(1,time):
            temp_i = []
            for i in ift_nodes:
                hpes = GetHpe(i, matrix)
                nbnodes = GetNodesofHpes(hpes, matrix)
                temp_i.extend(c for c in ContactNodes(nbnodes, beta) if c not in temp_i and c not in ift_nodes)
            ift_nodes.extend(i for i in temp_i)
            spt_nodes = [s for s in range(N) if s not in ift_nodes]
            ift_rates.iloc[init_node, t] = len(ift_nodes) / N
            spt_rates.iloc[init_node, t] = len(spt_nodes) / N

    return ift_rates.mean(axis=0), spt_rates.mean(axis=0)


def SIRCP(N, e, time, beta, gamma):
    matrix = GenerateHg(N, e)
    ift_rates = pd.DataFrame(np.zeros((N, time)))
    spt_rates = pd.DataFrame(np.zeros((N, time)))
    rmd_rates = pd.DataFrame(np.zeros((N, time)))
    ift_rates.iloc[:, 0] = 1/N
    spt_rates.iloc[:, 0] = 1-1/N

    for init_node in tqdm(range(N), desc="Loading..."):
        ift_nodes = [init_node]
        spt_nodes = [s for s in range(N) if s not in ift_nodes]
        rmd_nodes = []
        for t in range(time):
            temp_i = []
            for i in ift_nodes:
                hpes = GetHpe(i, matrix)
                hpe = ChooseHpe(hpes)
                nbnodes = GetNodesofHpe(hpe, matrix)
                temp_i.extend(c for c in ContactNodes(nbnodes, beta) if c not in temp_i and c in spt_nodes) # 去重、去I、去R

            rmd_nodes.extend(RemoveNode(ift_nodes, gamma))
            ift_nodes.extend(temp_i)
            ift_nodes = [i for i in ift_nodes if i not in rmd_nodes]
            spt_nodes = [s for s in range(N) if s not in ift_nodes and s not in rmd_nodes]

            ift_rates.iloc[init_node, t] = len(ift_nodes) / N
            spt_rates.iloc[init_node, t] = len(spt_nodes) / N
            rmd_rates.iloc[init_node, t] = len(rmd_nodes) / N

    return ift_rates.mean(axis=0), spt_rates.mean(axis=0), rmd_rates.mean(axis=0)

beta = 0.02
gamma = 0.1
# ift_rate, spt_rate = SICP(100, 10, 25, 0.02)
# ift_rate, spt_rate = SIRP(100, 10, 25, 0.02)
ift_rate, spt_rate, rmd_rate = SIRCP(100, 10, 50, beta, gamma)
plt.title('SIR ON HG (CP); beta=%.2f, gamma=%.2f'%(beta,gamma))

plt.plot(ift_rate, label='i(t)', color='#252a34')
plt.plot(spt_rate, label='s(t)', color='#ff2e63')
plt.plot(rmd_rate, label='r(t)')

plt.legend()
plt.show()


