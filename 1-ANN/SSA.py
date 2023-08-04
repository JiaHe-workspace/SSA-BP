# 麻雀搜索算法，随机优化

import copy
import random
import numpy as np
import os
import scipy.io
from matplotlib import pyplot as plt

''' Tent种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    round = np.random.rand()
    for i in range(pop):
        for j in range(dim):
            round = 4*round*(1-round)  # Logistic混沌映射
            X[i, j] = round * (ub[j] - lb[j]) + lb[j]
    return X, lb, ub


'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]  # 超过边界后随机
                X[i, j] = ub[j]  # 超过边界后极值替换
            elif X[i, j] < lb[j]:
                # X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''
def CaculateFitness(X, fun, dim):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    # fitAvg = fitness.sum() / fitness.shape[0]
    # x1 = np.zeros([1, dim])
    # x1 = np.squeeze(x1)
    # round = np.random.rand()
    # for i in range(pop):
    #     if fitness[i] < fitAvg:
    #         for j in range(dim):
    #             x1[j] = X[i, j] * (random.gauss(0, 1) + 1)
    #         funx1 = fun(x1)
    #         if funx1 < fitness[i]:
    #             X[i] = x1
    #             fitness[i] = funx1
    #     else:
    #         for j in range(dim):
    #             round = 4 * round * (1 - round)  # Logistic混沌映射
    #             x1[j] = (X[i, j] + round) / 2
    #         funx1 = fun(x1)
    #         if funx1 < fitness[i]:
    #             X[i] = x1
    #             fitness[i] = funx1
    return fitness, X


'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)  # 按列升序
    index = np.argsort(Fit, axis=0)  # 按列升序，返回排序后的索引值的数组
    return fitness, index


'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''麻雀发现者勘探更新'''
def PDUpdate(X, PDNumber, ST, Max_iter, dim):
    X_new = copy.copy(X)
    sta = 0.2 * PDNumber

    for p in range(PDNumber):
        R2 = np.random.rand(1)  # 返回随机生成的一个实数，它在[0,1)范围内
        for j in range(dim):
            if R2 < ST or p < 1:
                X_new[p, j] = X[p, j] * np.exp(-p / (random.random() * Max_iter))
            else:
                X_new[p, j] = X[p, j] + random.gauss(0, 1)   # random.gauss(0, 1)服从正态分布的均值为0，标准差为1的随机数
    return X_new


'''麻雀加入者更新'''
def JDUpdate(X, PDNumber, pop, dim):
    X_new = copy.copy(X)
    # 产生-1，1的随机数
    A = np.ones([dim, 1])
    round = np.random.rand()
    for a in range(dim):
        round = 4 * round * (1 - round)
        if (round > 0.5):
            A[a] = -1
    aa = np.linalg.inv(np.matrix(A.T) * np.matrix(A))
    for i in range(PDNumber + 1, pop):
        for j in range(dim):
            round = 4 * round * (1 - round)
            if i > (pop - PDNumber) / 2 + PDNumber:
                # 第i个加入者适应度较低且没有获得食物，处于十分饥饿的状态，需要飞往其它区域以补充能量
                X_new[i, j] = random.gauss(0, 1) * np.exp((X[-1, j] - X[i, j]) / i ** 2)
            else:
                # 当i<0.5n时，第i个加入者将在Xp附近随机觅食,此处Xp=X0
                AA = np.mean(np.abs(X[i, j] - X[0, :]) * A * np.array(aa))
                X_new[i, j] = X[0, j] - AA
    return X_new


'''警戒者更新'''
def SDUpdate(X, pop, SDNumber, fitness, BestF):
    X_new = copy.copy(X)
    dim = X.shape[1]
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)  # 返回一个长度为pop新列表，新列表存放Temp所产生pop个随机唯一的元素
    SDchooseIndex = RandIndex[0:SDNumber]  # 获取紧接着坐标
    for i in range(SDNumber):
        for j in range(dim):
            if fitness[SDchooseIndex[i]] > BestF:
                # 该麻雀处于种群的边缘，容易受到捕食者的攻击, 会靠近或远离最优点
                X_new[SDchooseIndex[i], j] = X[0, j] + random.gauss(0, 1) * np.abs(X[SDchooseIndex[i], j] - X[0, j])
            elif fitness[SDchooseIndex[i]] == BestF:
                # 处于种群中间的麻雀意识到了危险，需要接近种群中其它麻雀以降低被捕食的概率
                K = 4 * random.random() - 2  # K 在这里表示麻雀移动的方向，同时也是步长控制参数
                X_new[SDchooseIndex[i], j] = X[SDchooseIndex[i], j] + K * (
                        np.abs(X[SDchooseIndex[i], j] - X[-1, j]) / (fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
    return X_new


'''麻雀搜索算法'''
def Tent_SSA(pop, dim, lb, ub, Max_iter, fun):
    '''
    输入：pop=>麻雀个体数量； dim=>目标函数变量空间的维数  lb=>下边界 ub=>上边界  fun=>适应度计算函数  Max_iter=>最大迭代次数
    返回：GbestScore=>全局最优适应度, GbestPositon=>最优参数, Curve=>全局最优适应度变化nparray
    '''
    ST = 0.6  # 预警值 [0.5,1]
    PD = 0.8  # 发现者的比列，剩下的是加入者
    SD = 0.3  # 意识到有危险麻雀的比重
    PDNumber = int(pop * PD)  # 发现者数量
    SDNumber = int(pop * SD)  # 意识到有危险麻雀数量
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness, X = CaculateFitness(X, fun, dim)  # 计算适应度值
    fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
    X = SortPosition(X, sortIndex)  # 种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[0, :])
    Curve = np.zeros([Max_iter, 1])
    for i in range(Max_iter):
        BestF = fitness[0]

        X = PDUpdate(X, PDNumber, ST, Max_iter, dim)  # 发现者更新
        # X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = JDUpdate(X, PDNumber, pop, dim)  # 加入者更新
        # X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        X = SDUpdate(X, pop, SDNumber, fitness, BestF)  # 警戒者更新
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测

        fitness, X = CaculateFitness(X, fun, dim)  # 计算适应度值

        fitness, sortIndex = SortFitness(fitness)  # 对适应度值排序
        X = SortPosition(X, sortIndex)  # 种群排序
        if (fitness[0] <= GbestScore):  # 更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0, :] = copy.copy(X[0, :])
        Curve[i] = GbestScore  # 最优适应度
        if GbestScore[0] < 0.001:
            break
        # Curve[i] = ditAvg  # 平均适应度
    return GbestScore, GbestPositon, Curve

'''定义适应度函数'''
# def fun_cnn(X):
#     import rbf_ANN as ra
#     from sklearn import metrics
#
#     X_train, y_train, X_test, Y_test, dataall = ra.make_data()
#     X_train = np.array(X_train.values.tolist())
#     Y_train = np.array(y_train.values.tolist())
#
#     X = X[:]
#     X = X.tolist()
#
#     rbf = ra.RBFnetwork(hidden_nums=X[0], r_w=X[1], r_c=X[2], r_sigma=X[3], iters=1000)  # 网络初始化
#     rbf.train(X_train, Y_train)  # 训练模型
#     train_pred = rbf.predict(X_train)  # 预测样本
#
#     # 计算适应度（绝对平方差）
#     mae = round(metrics.mean_absolute_error(Y_train, train_pred), 2)
#     # mse = round(metrics.mean_squared_error(Y_train, train_pred), 2)
#     # rmse = round(np.sqrt(metrics.mean_squared_error(Y_train, train_pred)), 2)
#     # r2_square = round(metrics.r2_score(Y_train, train_pred), 2)
#
#     return mae

if __name__ ==  "__main__":
    # 设置参数
    pop = 20  # 种群数量
    Max_iter = 10  # 最大迭代次数
    dim = 4  # 维度
    lb = np.zeros((dim, 1))
    ub = np.zeros((dim, 1))
    lb[0] = 5 * np.ones([1, 1])  # 隐藏节点数目下限
    ub[0] = 30 * np.ones([1, 1])  # 隐藏节点数目上限
    lb[1] = 0.01 * np.ones([1, 1])  # r_w下边界
    ub[1] = 30.9 * np.ones([1, 1])  # 上边界
    lb[2] = 30.1 * np.ones([1, 1])  # r_c下边界
    ub[2] = 45.9 * np.ones([1, 1])  # 上边界
    lb[3] = 45.1 * np.ones([1, 1])  # r_sigma下边界
    ub[3] = 60.9 * np.ones([1, 1])  # 上边界

    # fun = fun_cnn
    # GbestScore, GbestPositon, Curve = Tent_SSA(pop, dim, lb, ub, Max_iter, fun)

    # 绘制适应度曲线
    plt.figure(1)
    # plt.plot(100 * (1 - Curve), 'r-', linewidth=2)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel("Fitness", fontsize='medium')
    plt.grid()
    plt.title('SSA', fontsize='large')
    plt.show()

