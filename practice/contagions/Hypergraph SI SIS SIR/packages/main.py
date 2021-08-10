from main_func import Main

if __name__ == '__main__':
    """
    :param theta: 参数列表
    :param iters: 迭代次数
    :param strategy: 策略
    :param type: 传播类型
    :return: void
    """
    Main.startMain([0.2, 0.1], 25, 'RP', 'SIR')