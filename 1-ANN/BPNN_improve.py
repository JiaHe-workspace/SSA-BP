# coding=utf-8
# 作者：Liu Jiahe；
# 功能：改进BP神经网络拟合；
# 时间：2023 年 7 月 22 日；
# 备注：无。

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import random

def zongchazhi(y_test, y_test_pred):
    '''计算实际结果与预测结果的总差值,偏差率'''
    aaa = y_test
    bbb = y_test_pred
    sum1 = sum(aaa)
    sum2 = sum(aaa) - sum(bbb)
    rate = sum2 / sum1
    print(f'总差值：{round(sum2, 3)}, 偏差率：{round(rate * 100, 2)}%',end='   ')
    return round(sum2, 3), round(rate * 100, 2)

def print_evaluate(true, predicted):
    '''预测结果评价'''
    mae = round(metrics.mean_absolute_error(true, predicted), 3)
    mse = round(metrics.mean_squared_error(true, predicted), 3)
    rmse = round(np.sqrt(metrics.mean_squared_error(true, predicted)), 3)
    r2_square = round(metrics.r2_score(true, predicted), 3)
    error, errate = zongchazhi(true, predicted)
    print('MAE:', mae,end='   ')
    print('MSE:', mse,end='   ')
    print('RMSE:', rmse,end='   ')
    print('R2 Square:', r2_square)
    # print('__________________________________')
    return [mae, mse, rmse, r2_square, error, errate]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_data():
    # 加载数据集
    df = pd.read_excel('E:\learn\zhanyonglv\cost_deal\cost_data_6m.xlsx', sheet_name='data1')
    df = df.sort_values(by=["OPERATION_END_TIME"], ascending=[True])
    df.index = range(len(df))
    df = df[['NUMBERES', 'TIME_CYCLE', 'HEAT_START_WEIGHT', 'PROCESS_COST', "TON_STEEL_COST", 'cost_zzfy',
             'cost_hj', 'cost_fg', 'cost_gtl', 'cost_fzcl', 'cost_yfl', 'cost_nc', 'cost_nydl', 'cost_rg']]
    df = df.fillna(-1)

    # 剔除成本差异太大的样本
    avg = df['TON_STEEL_COST'].sum() / df['TON_STEEL_COST'].shape[0]
    for i in range(0, df.shape[0]):
        if abs(df['TON_STEEL_COST'][i] - avg) > avg * 0.5:
            df = df.drop(index=[i])
    df.index = range(len(df))

    # 查看数据项
    features = df[['NUMBERES', 'TIME_CYCLE', 'HEAT_START_WEIGHT', 'cost_zzfy',
                   'cost_gtl', 'cost_fzcl', 'cost_nc', 'cost_rg','cost_hj']]  #, 'cost_nydl'
    target = df[["TON_STEEL_COST"]]
    # dataall = df[['NUMBERES', 'TIME_CYCLE', 'HEAT_START_WEIGHT', 'cost_zzfy', 'cost_hj',
    #               'cost_gtl', 'cost_fzcl', 'cost_nc', 'cost_nydl', 'cost_rg', "PROCESS_COST"]]

    # ##数据归一化处理
    # min_max_scaler = preprocessing.MinMaxScaler()
    # features = min_max_scaler.fit_transform(features)

    # # 数据集划分
    # split_num = int(len(features) * 0.5)
    # X_train = features[:split_num]
    # Y_train = target[:split_num]
    # X_test = features[split_num:]
    # Y_test = target[split_num:]
    # return X_train, Y_train, X_test, Y_test, features, target

    # 将输入数据改为矩阵；
    samplein = np.mat(features).T

    # 将输出数据改为矩阵；
    sampleout = np.mat(target).T

    # 求得输入数据的最小值和最大值：[最小值 最大值]；

    sampleinminmax = np.array(
        [samplein.min(axis=1).T.tolist()[0],
         samplein.max(axis=1).T.tolist()[0]]
    ).transpose()
    # print("sampleinminmax: ", sampleinminmax)
    # [[20.55 60.63]
    #  [ 0.6   3.1 ]
    #  [ 0.09  0.79]]

    # 求得 passengertraffic 和 freighttraffic 的最小值和最大值：[最小值 最大值]；
    sampleoutminmax = np.array(
        [sampleout.min(axis=1).T.tolist()[0],
         sampleout.max(axis=1).T.tolist()[0]]
    ).transpose()

    # 将数据进行标准化（归一化）；

    # 将数据映射到 -1 到 1 的范围内；
    sampleinnorm = (
            2 *
            (np.array(samplein.T) - sampleinminmax.transpose()[0]) /
            (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0])
            - 1
    ).transpose()

    # 将数据映射到 -1 到 1 的范围内；
    sampleoutnorm = (
            2 *
            (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) /
            (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0])
            - 1
    ).transpose()

    # 返回标准化后的输入、输出数据、原始数据的最值；
    return sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout

def make_data1():
    # 加载数据集, 单一钢种时间序列数据
    df = pd.read_excel('E:\learn\zhanyonglv\cost_deal\cost_data_S355-J1.xlsx', sheet_name='data1')
    df = df.sort_values(by=["OPERATION_END_TIME"], ascending=[True])
    df.index = range(len(df))
    df = df[['NUMBERES', 'TIME_CYCLE', 'HEAT_START_WEIGHT', 'PROCESS_COST', "TON_STEEL_COST", 'cost_zzfy',
             'cost_hj', 'cost_fg', 'cost_gtl', 'cost_fzcl', 'cost_yfl', 'cost_nc', 'cost_nydl', 'cost_rg']]
    df = df.fillna(-1)

    # 剔除成本差异太大的样本
    avg = df['TON_STEEL_COST'].sum() / df['TON_STEEL_COST'].shape[0]
    for i in range(0, df.shape[0]):
        if abs(df['TON_STEEL_COST'][i] - avg) > avg * 0.5:
            df = df.drop(index=[i])
    df.index = range(len(df))

    len1 = 10
    dataTime = pd.DataFrame(columns=range(len1 + 1))
    for i in range(0, df.shape[0]-(len1 + 1)):
        dataTime.loc[i] = [0] * (len1 + 1)
        for j in range(0, len1 + 1):
            dataTime.loc[i, j] = df['TON_STEEL_COST'][i + j]

    # 查看数据项
    features = dataTime[range(len1)].astype('float')
    target = dataTime[[10]].astype('float')
    # dataall = df[['NUMBERES', 'TIME_CYCLE', 'HEAT_START_WEIGHT', 'cost_zzfy', 'cost_hj',
    #               'cost_gtl', 'cost_fzcl', 'cost_nc', 'cost_nydl', 'cost_rg', "PROCESS_COST"]]

    # ##数据归一化处理
    # min_max_scaler = preprocessing.MinMaxScaler()
    # features = min_max_scaler.fit_transform(features)

    # # 数据集划分
    # split_num = int(len(features) * 0.5)
    # X_train = features[:split_num]
    # Y_train = target[:split_num]
    # X_test = features[split_num:]
    # Y_test = target[split_num:]
    # return X_train, Y_train, X_test, Y_test, features, target

    # 将输入数据改为矩阵；
    samplein = np.mat(features).T

    # 将输出数据改为矩阵；
    sampleout = np.mat(target).T

    # 求得输入数据的最小值和最大值：[最小值 最大值]；

    sampleinminmax = np.array(
        [samplein.min(axis=1).T.tolist()[0],
         samplein.max(axis=1).T.tolist()[0]]
    ).transpose()
    # print("sampleinminmax: ", sampleinminmax)
    # [[20.55 60.63]
    #  [ 0.6   3.1 ]
    #  [ 0.09  0.79]]

    # 求得 passengertraffic 和 freighttraffic 的最小值和最大值：[最小值 最大值]；
    sampleoutminmax = np.array(
        [sampleout.min(axis=1).T.tolist()[0],
         sampleout.max(axis=1).T.tolist()[0]]
    ).transpose()

    # 将数据进行标准化（归一化）；

    # 将数据映射到 -1 到 1 的范围内；
    sampleinnorm = (
            2 *
            (np.array(samplein.T) - sampleinminmax.transpose()[0]) /
            (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0])
            - 1
    ).transpose()

    # 将数据映射到 -1 到 1 的范围内；
    sampleoutnorm = (
            2 *
            (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) /
            (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0])
            - 1
    ).transpose()

    # 返回标准化后的输入、输出数据、原始数据的最值；
    return sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout

def normalizeData1(x, y):
    '''将数据进行标准化处理'''
    # global sampleout, sampleinminmax, sampleoutminmax

    # 将输入数据改为矩阵；
    samplein = np.mat(x).T

    # 将输出数据改为矩阵；
    sampleout = np.mat(y).T

    # 求得输入数据的最小值和最大值：[最小值 最大值]；


    sampleinminmax = np.array(
        [samplein.min(axis=1).T.tolist()[0],
         samplein.max(axis=1).T.tolist()[0]]
    ).transpose()
    # print("sampleinminmax: ", sampleinminmax)
    # [[20.55 60.63]
    #  [ 0.6   3.1 ]
    #  [ 0.09  0.79]]

    # 求得 passengertraffic 和 freighttraffic 的最小值和最大值：[最小值 最大值]；
    sampleoutminmax = np.array(
        [sampleout.min(axis=1).T.tolist()[0],
         sampleout.max(axis=1).T.tolist()[0]]
    ).transpose()


    # 将数据进行标准化（归一化）；

    # 将数据映射到 -1 到 1 的范围内；
    sampleinnorm = (
            2 *
            (np.array(samplein.T) - sampleinminmax.transpose()[0]) /
            (sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0])
            - 1
    ).transpose()


    # 将数据映射到 -1 到 1 的范围内；
    sampleoutnorm = (
            2 *
            (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) /
            (sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0])
            - 1
    ).transpose()


    # 返回标准化后的输入、输出数据、原始数据的最值；
    return sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout


def network1(sampleinnorm, sampleoutnorm):
    '''神经网络；'''
    # global sampleoutminmax, errhistory

    # 初始化神经网络的参数；

    # 训练的次数；
    maxepochs = 10000
    # 学习率；
    learnrate = 0.0006
    # 认为可以停止训练的理想误差，达到该误差值时停止训练；
    errorfinal = 10**-2
    # 样本数，下面反向求解计算时用于产生样本数个 1 参与运算，用于计数；
    # samnum = 20
    samnum = sampleinnorm.shape[1]
    # 输入神经网络的数据维度；
    # indim = 3
    indim = sampleinnorm.shape[0]
    # 输出神经网络的数据维度；
    # outdim = 2
    outdim = sampleoutnorm.shape[0]
    # 隐藏层的节点数；
    hiddenunitnum = 8
    hiddenunitnum1 = 6

    # 使用 np.random.rand(3, 1) 创建一个数据范围在 [0, 1] 之间的 3 行 1 列的随机矩阵；
    # 通过 2 * np.random.rand(3, 1) - 1 的方法将数据范围映射到 [-1, 1] 之间；

    # 创建隐藏层的 w 权重和 b 偏置矩阵；
    # w1 = 2 * np.random.rand(hiddenunitnum, indim) - 1
    # b1 = 2 * np.random.rand(hiddenunitnum, 1) - 1
    w1 = 2 * np.random.rand(hiddenunitnum, indim) - 1
    b1 = 2 * np.random.rand(hiddenunitnum, 1) - 1

    # 创建隐藏层1的 w 权重和 b 偏置矩阵；
    w11 = 2 * np.random.rand(hiddenunitnum1, hiddenunitnum) - 1
    b11 = 2 * np.random.rand(hiddenunitnum1, 1) - 1

    # 创建输出层的 w 权重和 b 偏置矩阵；
    w2 = 2 * np.random.rand(outdim, hiddenunitnum1) - 1
    b2 = 2 * np.random.rand(outdim, 1) - 1

    # 创建一个列表，用于存储每次训练产生的误差，训练结束后用于绘图，进行可视化，便于分析整个训练过程误差的变化情况；
    errhistory = []

    # for j in range(maxepochs):  # 随机梯度下降
    #     dataIndex = list(range(samnum))
    #     for i in range(samnum):
    #         learnrate = 1/(1 + i +j) + 0.000001
    #         randIndex = int(random.uniform(0, len(dataIndex)))
    #
    #         # 通过 h = sigmoid(w1 x + b1) 计算隐藏层的输出；  np.dot点积   transpose()默认是矩阵转置，指定时轴转换
    #         hiddenout = sigmoid((np.dot(w1, sampleinnorm[:, randIndex]).transpose() + b1.transpose())).transpose()
    #
    #         # 通过 y = w2 h + b2 计算输出层的输出；
    #         # 因为我们希望使用神经网络进行多输入参数的拟合，不是解决分类问题，所以输出层不能使用激活函数进行非线性化处理；
    #         networkout = (np.dot(w2, hiddenout).transpose() + b2.transpose()).transpose()
    #
    #         # 计算神经网络输出和真实输出之间的误差；
    #         err = sampleoutnorm[:, randIndex] - networkout
    #
    #         # 使用一个列表将误差存储下来；
    #         errhistory.append(err[0][0])
    #
    #         # 如果误差已经在期望的范围内，则停止训练；
    #         if abs(err) < errorfinal:
    #             break
    #
    #         # 开始进行反向传递；
    #
    #         # 因为是使用神经网络进行拟合，所以最终的输出层不适用激活函数进行非线性化；
    #         # 所以最终额输出层的损失函数不需要对最终输出求偏导，直接赋值即可；
    #         delta2 = err
    #
    #         # 隐藏层的损失函数对隐藏层的输出求偏导；
    #         # d_L_d_ypred × d_ypred_d_h
    #         # 使用 x = (w2 × (ytrue - ypred))f'(h) = (w2 × (ytrue - ypred))(h)(1 - h) 进行反向求解；
    #         delta1 = np.dot(w2.transpose(), delta2) * hiddenout * (1 - hiddenout)
    #
    #         # 计算输出层的权重 w2 和偏置 b2；
    #         # dw2 = (ytrue - ypred) × h
    #         dw2 = np.dot(delta2, hiddenout.transpose())
    #         # db2 = (ytrue - ypred) × 1
    #         db2 = np.dot(delta2, np.ones((1, 1)))
    #
    #         # 计算隐藏层的权重 w1 和偏置 b1；
    #         # dw1 = d_L_d_ypred × d_ypred_d_h  × x
    #         aa = sampleinnorm[:, randIndex].reshape(-1,1).transpose()
    #         dw1 = np.dot(delta1, sampleinnorm[:, randIndex].reshape(-1,1).transpose())
    #         # db1 = d_L_d_ypred × d_ypred_d_h × 1
    #         db1 = np.dot(delta1, np.ones((1, 1)))
    #
    #         # 更新输出层的权重 w2 和偏置 b2；
    #         w2 += learnrate * dw2
    #         b2 += learnrate * db2
    #
    #         # 更新隐藏层的权重 w1 和 b1；
    #         w1 += learnrate * dw1
    #         b1 += learnrate * db1
    #
    #         del dataIndex[randIndex]

    for _ in range(maxepochs):  # 批量梯度下降
        # 通过 h = sigmoid(w1 x + b1) 计算隐藏层的输出；  np.dot点积   transpose()默认是矩阵转置，指定时轴转换
        hiddenout = sigmoid((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()

        hiddenout1 = sigmoid((np.dot(w11, hiddenout).transpose() + b11.transpose())).transpose()
        # 通过 y = w2 h + b2 计算输出层的输出；
        # 因为我们希望使用神经网络进行多输入参数的拟合，不是解决分类问题，所以输出层不能使用激活函数进行非线性化处理；
        networkout = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()

        # 计算神经网络输出和真实输出之间的误差；
        err = sampleoutnorm - networkout

        # 对误差进行求和，得到总的误差；
        # 使用整个数据集的误差和作为误差，而不是单条数据段，目的是为了更快的去降低误差；
        sse = sum(sum(err ** 2))

        # 使用一个列表将误差存储下来；
        errhistory.append(sse)

        # 如果误差已经在期望的范围内，则停止训练；
        if sse < errorfinal:
            break

        # 开始进行反向传递；

        # 因为是使用神经网络进行拟合，所以最终的输出层不适用激活函数进行非线性化；
        # 所以最终额输出层的损失函数不需要对最终输出求偏导，直接赋值即可；
        delta2 = err

        # 隐藏层的损失函数对隐藏层的输出求偏导；
        # d_L_d_ypred × d_ypred_d_h
        # 使用 x = (w2 × (ytrue - ypred))f'(h) = w2 × (ytrue - ypred)(h)(1 - h) 进行反向求解；
        delta11 = np.dot(w2.transpose(), delta2) * hiddenout1 * (1 - hiddenout1)
        delta1 = np.dot(w11.transpose(), delta11) * hiddenout * (1 - hiddenout)


        # 计算输出层的权重 w2 和偏置 b2；
        # dw2 = (ytrue - ypred) × h
        dw2 = np.dot(delta2, hiddenout1.transpose())
        # db2 = (ytrue - ypred) × 1
        db2 = np.dot(delta2, np.ones((samnum, 1)))

        # 计算隐藏层的权重 w1 和偏置 b1；
        # dw1 = d_L_d_ypred × d_ypred_d_h  × x
        dw11 = np.dot(delta11, hiddenout.transpose())
        # db1 = d_L_d_ypred × d_ypred_d_h × 1
        db11 = np.dot(delta11, np.ones((samnum, 1)))

        # 计算隐藏层1的权重 w1 和偏置 b1；
        dw1 = np.dot(delta1, sampleinnorm.transpose())
        db1 = np.dot(delta1, np.ones((samnum, 1)))

        # 更新输出层的权重 w2 和偏置 b2；
        w2 += learnrate * dw2
        b2 += learnrate * db2

        # 更新隐藏层的权重 w11 和 b11；
        w11 += learnrate * dw11
        b11 += learnrate * db11

        # 更新隐藏层的权重 w1 和 b1；
        w1 += learnrate * dw1
        b1 += learnrate * db1

    return w1, b1, w11, b11, w2, b2, errhistory

def predict1(w1, b1, w11, b11, w2, b2, sampleinnorm):
    '''使用训练好的神经网络进行预测；'''
    # 使用新的权重计算隐藏层和输出层的输出；
    # h = f(w1 x + b1)
    # h1 = f(w11 h + b11)
    # o = w2 h + b2

    hiddenout = sigmoid((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
    hiddenout1 = sigmoid((np.dot(w11, hiddenout).transpose() + b11.transpose())).transpose()
    networkout = (np.dot(w2, hiddenout1).transpose() + b2.transpose()).transpose()
    return networkout


class BPNN:
    # 构造三层BP网络架构
    def __init__(self, hiddenunitnum=8, hiddenunitnum2=6, maxepochs=10000, learnrate=0.0006, findBest='SSA'):
        # 训练的次数；
        self.maxepochs = maxepochs
        # 学习率；
        self.learnrate = learnrate
        self.adadeltaRate = 0.6 # Adadelta衰减参数
        # 认为可以停止训练的理想误差，达到该误差值时停止训练；
        self.errorfinal = 10 ** -4
        # 搜索算法
        self.finfBest = findBest

        # # 输入层，隐藏层，输出层的节点数
        # self.indim = indim
        # self.hiddenunitnum = hiddenunitnum
        # self.outdim = outdim
        # 隐藏层的节点数；
        self.hiddenunitnum = hiddenunitnum
        self.hiddenunitnum2 = hiddenunitnum2

        # 创建一个列表，用于存储每次训练产生的误差，训练结束后用于绘图，进行可视化，便于分析整个训练过程误差的变化情况；
        self.errhistory = []
        self.deltaStroy = np.array([])  # 存储梯度

    def train(self, sampleinnorm, sampleoutnorm):
        self.sampleinnorm = sampleinnorm
        self.sampleoutnorm = sampleoutnorm
        # 样本数，下面反向求解计算时用于产生样本数个 1 参与运算，用于计数；
        self.samnum = sampleinnorm.shape[1]
        # 输入神经网络的数据维度；
        self.indim = sampleinnorm.shape[0]
        # 输出神经网络的数据维度；
        self.outdim = sampleoutnorm.shape[0]

        # 创建隐藏层的 w 权重和 b 偏置矩阵；
        self.w1 = 2 * np.random.rand(self.hiddenunitnum, self.indim) - 1
        self.b1 = 2 * np.random.rand(self.hiddenunitnum, 1) - 1

        # 创建隐藏层的 w 权重和 b 偏置矩阵；
        self.w2 = 2 * np.random.rand(self.hiddenunitnum2, self.hiddenunitnum) - 1
        self.b2 = 2 * np.random.rand(self.hiddenunitnum2, 1) - 1

        # 创建输出层的 w 权重和 b 偏置矩阵；
        self.w0 = 2 * np.random.rand(self.outdim, self.hiddenunitnum2) - 1
        self.b0 = 2 * np.random.rand(self.outdim, 1) - 1

        self.deltaStroy = np.array([[np.zeros([self.hiddenunitnum, self.indim]), np.zeros([self.hiddenunitnum, 1]),
                                     np.zeros([self.hiddenunitnum2, self.hiddenunitnum]), np.zeros([self.hiddenunitnum2, 1]),
                                     np.zeros([self.outdim, self.hiddenunitnum2]), np.zeros([self.outdim, 1])]], dtype=object)  # 存储梯度
        self.deltaWB = np.array([[np.zeros([self.hiddenunitnum, self.indim]), np.zeros([self.hiddenunitnum, 1]),
                                     np.zeros([self.hiddenunitnum2, self.hiddenunitnum]), np.zeros([self.hiddenunitnum2, 1]),
                                     np.zeros([self.outdim, self.hiddenunitnum2]), np.zeros([self.outdim, 1])]], dtype=object)  # 存储累加量

        if self.finfBest == 'SSA':
            self.errorbackpropagateSSA()  # 麻雀搜索
        elif self.finfBest == 'BGD':
            for i in range(self.maxepochs):
                self.predict2(sampleinnorm)  # 正向传播
                # self.errorbackpropagateBGD(sampleinnorm, sampleoutnorm)  # 反向传播，批量梯度下降BGD
                self.errorbackpropagateBGD_Adadelta(sampleinnorm, sampleoutnorm)  # Adadelta自适应调整梯度下降优化

        elif self.finfBest == 'SGD':
            self.maxepochs = 100
            self.learnrate = 0.6
            for j in range(self.maxepochs):  # 随机梯度下降SGD
                dataIndex = np.random.permutation(self.samnum)  # 洗牌
                # dataIndex = list(range(self.samnum))
                for i in range(self.samnum):
                    # randIndex = int(random.uniform(0, len(dataIndex)))
                    dataIn = sampleinnorm[:, i].reshape(-1, 1)
                    dataOut = sampleoutnorm[:, i].reshape(-1, 1)
                    self.predict2(dataIn)  # 正向传播
                    # self.errorbackpropagateBGD(dataIn, dataOut)  # 反向传播，随机梯度下降
                    self.errorbackpropagateBGD_Adadelta(dataIn, dataOut)  # Adadelta自适应调整梯度下降优化
                    # del dataIndex[randIndex]

        elif self.finfBest == 'MBGD':
            self.maxepochs = 100
            self.learnrate = 0.006
            minibatch_size = 20  # 对数据集遍历时的步长
            for j in range(self.maxepochs):  # 小批量梯度下降MBGD
                dataIndex = np.random.permutation(self.samnum)  # 洗牌
                sampleinnorm = self.sampleinnorm[:, dataIndex]
                sampleoutnorm = self.sampleoutnorm[:, dataIndex]
                for i in range(0, self.samnum, minibatch_size):
                    dataIn = sampleinnorm[:, i:i + minibatch_size]  # 切片
                    dataOut = sampleoutnorm[:, i:i + minibatch_size]
                    self.predict2(dataIn)  # 正向传播
                    # self.errorbackpropagateBGD(dataIn, dataOut)  # 反向传播，梯度下降
                    self.errorbackpropagateBGD_Adadelta(dataIn, dataOut)  # Adadelta自适应调整梯度下降优化

    def errorbackpropagateSSA(self,):
        # 麻雀搜索权值和阈值
        import SSA
        # 设置参数
        pop = 50  # 种群数量
        Max_iter = 1000  # 最大迭代次数
        dim = self.indim * self.hiddenunitnum + self.hiddenunitnum + \
              self.hiddenunitnum * self.hiddenunitnum2 + self.hiddenunitnum2 + \
              self.hiddenunitnum2 * self.outdim + self.outdim
        # 维度  dim=inputnum*hiddennum_best+hiddennum_best+hiddennum_best*outputnum+outputnum
        lb = np.zeros((dim, 1)) - 3
        ub = np.zeros((dim, 1)) + 3

        fun = self.fun_cnn
        GbestScore, GbestPositon, Curve = SSA.Tent_SSA(pop, dim, lb, ub, Max_iter, fun)
        self.XtoWB(GbestPositon.T)
        # 使用一个列表将误差存储下来；
        self.errhistory = Curve.tolist()

    def fun_cnn(self, X):
        '''
        定义适应度函数
        :param X权值、阈值:
        :return:适应度
        '''
        self.XtoWB(X)  # X还原阈值w1、w0;权值b1、b0
        networkout = self.predict2(self.sampleinnorm)  # 正向传播

        ## 适应度计算函数选择
        # sse = sum(sum((self.sampleoutnorm - networkout) ** 2))  # 对误差进行求和，得到总的误差；
        # sse = metrics.mean_absolute_error(self.sampleoutnorm, networkout)
        sse = metrics.mean_squared_error(self.sampleoutnorm, networkout)
        # sse = np.sqrt(metrics.mean_squared_error(self.sampleoutnorm, networkout))
        # sse = 1 - metrics.r2_score(self.sampleoutnorm.tolist()[0], networkout.tolist()[0])
        return sse

    def errorbackpropagateBGD(self, sampleinnorm, sampleoutnorm):
        '''误差反向传播'''
        yangbenNum = sampleinnorm.shape[1]

        # 计算神经网络输出和真实输出之间的误差；
        err = sampleoutnorm - self.networkout

        # 对误差进行求和，得到总的误差；
        # 使用整个数据集的误差和作为误差，而不是单条数据段，目的是为了更快的去降低误差；
        sse = sum(sum(err ** 2))
        # sse = metrics.mean_squared_error(sampleoutnorm, self.networkout)

        # 使用一个列表将误差存储下来；
        self.errhistory.append(sse)

        # 如果误差已经在期望的范围内，则停止训练；
        if sse < self.errorfinal:
            return

        # 开始进行反向传递；

        # 因为是使用神经网络进行拟合，所以最终的输出层不适用激活函数进行非线性化；
        # 所以最终额输出层的损失函数不需要对最终输出求偏导，直接赋值即可；
        delta0 = err

        # 隐藏层的损失函数对隐藏层的输出求偏导；
        # d_L_d_ypred × d_ypred_d_h
        # 使用 x = (w0 × (ytrue - ypred))f'(h) = (w0 × (ytrue - ypred))(h)(1 - h) 进行反向求解；
        delta2 = np.dot(self.w0.transpose(), delta0) * self.hiddenout2 * (1 - self.hiddenout2)
        delta1 = np.dot(self.w2.transpose(), delta2) * self.hiddenout * (1 - self.hiddenout)

        # 计算输出层的权重 w0 和偏置 b0；
        # dw0 = (ytrue - ypred) × h
        dw0 = np.dot(delta0, self.hiddenout2.transpose())
        # db0 = (ytrue - ypred) × 1
        db0 = np.dot(delta0, np.ones((yangbenNum, 1)))

        # 计算隐藏层的权重 w2 和偏置 b2；
        # dw1 = d_L_d_ypred × d_ypred_d_h  × x
        dw2 = np.dot(delta2, self.hiddenout.transpose())
        # db1 = d_L_d_ypred × d_ypred_d_h × 1
        db2 = np.dot(delta2, np.ones((yangbenNum, 1)))

        # 计算隐藏层的权重 w1 和偏置 b1；
        # dw1 = d_L_d_ypred × d_ypred_d_h  × x
        dw1 = np.dot(delta1, sampleinnorm.transpose())
        # db1 = d_L_d_ypred × d_ypred_d_h × 1
        db1 = np.dot(delta1, np.ones((yangbenNum, 1)))

        # 更新输出层的权重 w0 和偏置 b0；
        self.w0 += self.learnrate * dw0
        self.b0 += self.learnrate * db0

        # 更新隐藏层的权重 w2 和 b2；
        self.w2 += self.learnrate * dw2
        self.b2 += self.learnrate * db2

        # 更新隐藏层的权重 w1 和 b1；
        self.w1 += self.learnrate * dw1
        self.b1 += self.learnrate * db1


    def errorbackpropagateBGD_Adadelta(self, sampleinnorm, sampleoutnorm):
        '''误差反向传播'''
        yangbenNum = sampleinnorm.shape[1]

        # 计算神经网络输出和真实输出之间的误差；
        err = sampleoutnorm - self.networkout

        # 对误差进行求和，得到总的误差；
        # 使用整个数据集的误差和作为误差，而不是单条数据段，目的是为了更快的去降低误差；
        # sse = sum(sum(err ** 2))
        sse = metrics.mean_squared_error(sampleoutnorm, self.networkout)

        # 使用一个列表将误差存储下来；
        self.errhistory.append(sse)

        # 如果误差已经在期望的范围内，则停止训练；
        if sse < self.errorfinal:
            return

        # 开始进行反向传递；

        # 因为是使用神经网络进行拟合，所以最终的输出层不适用激活函数进行非线性化；
        # 所以最终额输出层的损失函数不需要对最终输出求偏导，直接赋值即可；
        delta0 = err

        # 隐藏层的损失函数对隐藏层的输出求偏导；
        # d_L_d_ypred × d_ypred_d_h
        # 使用 x = (w0 × (ytrue - ypred))f'(h) = (w0 × (ytrue - ypred))(h)(1 - h) 进行反向求解；
        delta2 = np.dot(self.w0.transpose(), delta0) * self.hiddenout2 * (1 - self.hiddenout2)
        delta1 = np.dot(self.w2.transpose(), delta2) * self.hiddenout * (1 - self.hiddenout)

        # 计算输出层的权重 w0 和偏置 b0；
        # dw0 = (ytrue - ypred) × h
        dw0 = np.dot(delta0, self.hiddenout2.transpose())
        # db0 = (ytrue - ypred) × 1
        db0 = np.dot(delta0, np.ones((yangbenNum, 1)))

        # 计算隐藏层的权重 w2 和偏置 b2；
        # dw1 = d_L_d_ypred × d_ypred_d_h  × x
        dw2 = np.dot(delta2, self.hiddenout.transpose())
        # db1 = d_L_d_ypred × d_ypred_d_h × 1
        db2 = np.dot(delta2, np.ones((yangbenNum, 1)))

        # 计算隐藏层的权重 w1 和偏置 b1；
        # dw1 = d_L_d_ypred × d_ypred_d_h  × x
        dw1 = np.dot(delta1, sampleinnorm.transpose())
        # db1 = d_L_d_ypred × d_ypred_d_h × 1
        db1 = np.dot(delta1, np.ones((yangbenNum, 1)))

        if len(self.errhistory) > 1:
            # 获取一定范围梯度的平方均值E[g**2]t
            ew0 = (self.deltaStroy[:, 4]**2).sum() / self.deltaStroy.shape[0]
            eb0 = (self.deltaStroy[:, 5]**2).sum() / self.deltaStroy.shape[0]
            ew2 = (self.deltaStroy[:, 2]**2).sum() / self.deltaStroy.shape[0]
            eb2 = (self.deltaStroy[:, 3]**2).sum() / self.deltaStroy.shape[0]
            ew1 = (self.deltaStroy[:, 0]**2).sum() / self.deltaStroy.shape[0]
            eb1 = (self.deltaStroy[:, 1]**2).sum() / self.deltaStroy.shape[0]
            # 获取一定范围累加量的平方均值E[dw**2]t
            addw0 = (self.deltaWB[:-1, 4] ** 2).sum() / self.deltaWB.shape[0]
            addb0 = (self.deltaWB[:-1, 5] ** 2).sum() / self.deltaWB.shape[0]
            addw2 = (self.deltaWB[:-1, 2] ** 2).sum() / self.deltaWB.shape[0]
            addb2 = (self.deltaWB[:-1, 3] ** 2).sum() / self.deltaWB.shape[0]
            addw1 = (self.deltaWB[:-1, 0] ** 2).sum() / self.deltaWB.shape[0]
            addb1 = (self.deltaWB[:-1, 1] ** 2).sum() / self.deltaWB.shape[0]

            # 求dwt = -( sqrt(E[dw(t-1)**2)](t-1) + 10E-8 ) * gt / ( sqrt(E[g**2](t-1)**2) + 10E-8 )      gt = 上面求导的dw
            edw0 = dw0 * np.sqrt((self.adadeltaRate * addw0 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 4]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * ew0 + (1 - self.adadeltaRate) * (dw0**2)) + 10E-8)
            edb0 = db0 * np.sqrt((self.adadeltaRate * addb0 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 5]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * eb0 + (1 - self.adadeltaRate) * (db0**2)) + 10E-8)
            edw2 = dw2 * np.sqrt((self.adadeltaRate * addw2 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 2]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * ew2 + (1 - self.adadeltaRate) * (dw2**2)) + 10E-8)
            edb2 = db2 * np.sqrt((self.adadeltaRate * addb2 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 3]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * eb2 + (1 - self.adadeltaRate) * (db2**2)) + 10E-8)
            edw1 = dw1 * np.sqrt((self.adadeltaRate * addw1 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 0]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * ew1 + (1 - self.adadeltaRate) * (dw1**2)) + 10E-8)
            edb1 = db1 * np.sqrt((self.adadeltaRate * addb1 + (1 - self.adadeltaRate) * (self.deltaWB[-1, 1]**2)) + 10E-8) / \
                   np.sqrt((self.adadeltaRate * eb1 + (1 - self.adadeltaRate) * (db1**2)) + 10E-8)

            self.deltaWB = np.append(self.deltaWB, [[edw1, edb1, edw2, edb2, edw0, edb0]], axis=0)

            # 更新输出层的权重 w0 和偏置 b0；
            self.w0 += edw0
            self.b0 += edb0

            # 更新隐藏层的权重 w2 和 b2；
            self.w2 += edw2
            self.b2 += edb2

            # 更新隐藏层的权重 w1 和 b1；
            self.w1 += edw1
            self.b1 += edb1
        else:
            # 更新输出层的权重 w0 和偏置 b0；
            self.w0 += self.learnrate * dw0
            self.b0 += self.learnrate * db0

            # 更新隐藏层的权重 w2 和 b2；
            self.w2 += self.learnrate * dw2
            self.b2 += self.learnrate * db2

            # 更新隐藏层的权重 w1 和 b1；
            self.w1 += self.learnrate * dw1
            self.b1 += self.learnrate * db1

        self.deltaStroy = np.append(self.deltaStroy, [[dw1, db1, dw2, db2, dw0, db0]], axis=0)
        if self.deltaStroy.shape[0] == 20:
            self.deltaStroy = np.delete(self.deltaStroy, 0, axis=0)
            self.deltaWB = np.delete(self.deltaWB, 0, axis=0)

    def predict2(self, sampleinnorm):
        '''使用训练好的神经网络进行预测；'''
        # 使用新的权重计算隐藏层和输出层的输出；
        # h = f(w1 x + b1)
        self.hiddenout = sigmoid((np.dot(self.w1, sampleinnorm).transpose() + self.b1.transpose())).transpose()
        # h2 = f(w2 x + b2)
        self.hiddenout2 = sigmoid((np.dot(self.w2, self.hiddenout).transpose() + self.b2.transpose())).transpose()
        # o = w0 h2 + b0
        self.networkout = (np.dot(self.w0, self.hiddenout2).transpose() + self.b0.transpose()).transpose()
        return self.networkout

    def XtoWB(self, X):
        '''将X拆解成w1、w0、b1、b0'''
        X = X[:]
        X = np.array(X)
        self.w1 = X[:self.indim * self.hiddenunitnum].reshape(self.hiddenunitnum, self.indim)
        self.b1 = X[self.indim * self.hiddenunitnum:
                    self.indim * self.hiddenunitnum + self.hiddenunitnum
                  ].reshape(self.hiddenunitnum, 1)
        self.w2 = X[self.indim * self.hiddenunitnum + self.hiddenunitnum:
                    self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2
                  ].reshape(self.hiddenunitnum2, self.hiddenunitnum)
        self.b2 = X[self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2:
                    self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2
                    + self.hiddenunitnum2
                  ].reshape(self.hiddenunitnum2, 1)
        self.w0 = X[self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2
                    + self.hiddenunitnum2:
                    self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.outdim
                  ].reshape(self.outdim, self.hiddenunitnum2)
        self.b0 = X[self.indim * self.hiddenunitnum + self.hiddenunitnum + self.hiddenunitnum * self.hiddenunitnum2
                    + self.hiddenunitnum2 + self.hiddenunitnum2 * self.outdim:].reshape(self.outdim, 1)

def scalableData1(normalizeData, minmax):
    # 由于进行了归一化，用最小值和最大值计算原始输出；
    # 计算样本原始数据的变化范围：最大值 - 最小值；
    diff = minmax[:, 1] - minmax[:, 0]

    # 将新权重计算的输出映射到 [0, 1] 范围；
    srcData = (normalizeData + 1) / 2
    # 将神经网络输出的第 1 个参数还原到原始的数据范围；
    for i in range(0, minmax.shape[0]):
        srcData[i] = srcData[i] * diff[i] + minmax[i][0]

    # srcData[0] = srcData[0] * diff[0] + minmax[0][0]
    # # 将神经网络输出的第 2 个参数还原到原始的数据范围；
    # srcData[1] = srcData[1] * diff[1] + minmax[1][0]
    return srcData

def plotGraph1(sampleout, errhistory, networkPredict):
    '''结果可视化；'''
    # global sampleout, errhistory, networkPredict
    # 输出转为 numpy；
    sampleout = np.array(sampleout)

    # 绘制第 1 张图；
    if errhistory != []:
        plt.figure(1)
        plt.plot(errhistory, label="err")
        plt.legend(loc='upper left')
        plt.show()

    # 绘制第 2 张图；
    plt.figure(2)
    # plt.subplot(2, 1, 1)
    plt.plot(sampleout[0],
             color="blue",
             linewidth=1.5,
             linestyle="-",
             label=u"real passengertraffic")

    plt.plot(networkPredict[0],
             color="red",
             linewidth=1.5,
             linestyle="--",
             label=u"predict passengertraffic")
    plt.legend(loc='upper left')
    plt.show()
    # plt.draw()

def haveTry():
    # global sampleoutminmax, networkPredict

    # X_train, y_train, X_test, Y_test, dataall = make_data()
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()  # normalizeData1(X_train, y_train) # 数据获取与归一化
    # 训练集，测试集划分
    split_num = int(sampleinnorm.shape[1] * 0.8)
    X_train = sampleinnorm[:, :split_num]
    Y_train = sampleoutnorm[:, :split_num]
    X_test = sampleinnorm[:, split_num:]
    Y_test = sampleoutnorm[:, split_num:]
    # 将输入、输出、原始数据的最值传入神经网络，对神经网络进行训练；
    w1, b1, w11, b11, w2, b2, errhistory = network1(X_train, Y_train)  # network1(sampleinnorm, sampleoutnorm)
    # 向训练好的神经网络内传入参数进行预测；
    networkPredict = predict1(w1, b1, w11, b11, w2, b2, X_train)
    # 将预测好的结果缩放到正常的数据范围内；
    networkPredict = scalableData1(networkPredict, sampleoutminmax)
    # 结论可视化；
    # dt.print_evaluate(y_train, networkPredict)
    plotGraph1(sampleout[:, :split_num], errhistory, networkPredict)
    print('Train set evaluation:')
    print_evaluate(sampleout[:, :split_num].tolist()[0], networkPredict.tolist()[0])

    # 测试集测试
    textPredict = predict1(w1, b1, w11, b11, w2, b2, X_test)
    textPredict = scalableData1(textPredict, sampleoutminmax)
    plotGraph1(sampleout[:, split_num:], errhistory, textPredict)
    print('Text set evaluation:')
    print_evaluate(sampleout[:, split_num:].tolist()[0], textPredict.tolist()[0])
    print()

def haveTryBP():
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()  # normalizeData1(X_train, y_train) # 数据获取与归一化
    # 训练集，测试集划分
    split_num = int(sampleinnorm.shape[1] * 0.8)
    X_train = sampleinnorm[:, :split_num]
    Y_train = sampleoutnorm[:, :split_num]
    X_test = sampleinnorm[:, split_num:]
    Y_test = sampleoutnorm[:, split_num:]

    bpSSA = BPNN(findBest='SSA')
    bpBGD = BPNN(findBest='BGD')
    bpSGD = BPNN(findBest='SGD')
    bpMGD = BPNN(findBest='MBGD')
    bpSSA.train(X_train, Y_train)
    bpBGD.train(X_train, Y_train)
    bpSGD.train(X_train, Y_train)
    bpMGD.train(X_train, Y_train)
    PreSSA = bpSSA.predict2(X_train)
    PreBGD = bpBGD.predict2(X_train)
    PreSGD = bpSGD.predict2(X_train)
    PreMGD = bpMGD.predict2(X_train)
    TESTPreSSA = bpSSA.predict2(X_test)
    TESTPreBGD = bpBGD.predict2(X_test)
    TESTPreSGD = bpSGD.predict2(X_test)
    TESTPreMGD = bpMGD.predict2(X_test)

    Y_test = scalableData1(Y_test, sampleoutminmax)
    PreSSA = scalableData1(PreSSA, sampleoutminmax)
    PreBGD = scalableData1(PreBGD, sampleoutminmax)
    PreSGD = scalableData1(PreSGD, sampleoutminmax)
    PreMGD = scalableData1(PreMGD, sampleoutminmax)
    TESTPreSSA = scalableData1(TESTPreSSA, sampleoutminmax)
    TESTPreBGD = scalableData1(TESTPreBGD, sampleoutminmax)
    TESTPreSGD = scalableData1(TESTPreSGD, sampleoutminmax)
    TESTPreMGD = scalableData1(TESTPreMGD, sampleoutminmax)


    print('SSA Train set evaluation:', end=' ')
    print_evaluate(sampleout[:, :split_num].tolist()[0], PreSSA.tolist()[0])
    print('BGD Train set evaluation:', end=' ')
    print_evaluate(sampleout[:, :split_num].tolist()[0], PreBGD.tolist()[0])
    print('SGD Train set evaluation:', end=' ')
    print_evaluate(sampleout[:, :split_num].tolist()[0], PreSGD.tolist()[0])
    print('MGD Train set evaluation:', end=' ')
    print_evaluate(sampleout[:, :split_num].tolist()[0], PreMGD.tolist()[0])
    print('SSA Text  set evaluation:', end=' ')
    print_evaluate(sampleout[:, split_num:].tolist()[0], TESTPreSSA.tolist()[0])
    print('BGD Text  set evaluation:', end=' ')
    print_evaluate(sampleout[:, split_num:].tolist()[0], TESTPreBGD.tolist()[0])
    print('SGD Text  set evaluation:', end=' ')
    print_evaluate(sampleout[:, split_num:].tolist()[0], TESTPreSGD.tolist()[0])
    print('MGD Text  set evaluation:', end=' ')
    print_evaluate(sampleout[:, split_num:].tolist()[0], TESTPreMGD.tolist()[0])

    plt.figure(1)
    plt.plot(bpSSA.errhistory, 'r+', label="SSAerr")
    plt.plot(bpBGD.errhistory, 'y.', label="BGDerr")
    plt.plot(bpSGD.errhistory, 'g*', label="SGDerr")
    plt.plot(bpMGD.errhistory, 'b-', label="MGDerr")
    plt.legend(loc='upper left')
    plt.show()

    # 可视化部分
    import seaborn as sns
    sns.set(font_scale=1.2)
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', size=14)
    plt.plot(list(range(0, X_test.shape[1])), Y_test.tolist()[0], marker='.')
    plt.plot(list(range(0, X_test.shape[1])), TESTPreSSA.tolist()[0], marker='.')
    plt.plot(list(range(0, X_test.shape[1])), TESTPreBGD.tolist()[0], marker='.')
    plt.plot(list(range(0, X_test.shape[1])), TESTPreSGD.tolist()[0], marker='.')
    plt.plot(list(range(0, X_test.shape[1])), TESTPreMGD.tolist()[0], marker='.')
    plt.legend(['真实值', 'SSA-预测值', 'BGD-预测值', 'SGD-预测值', 'MGD-预测值'])
    plt.title('转炉炼钢炉次成本预测')
    plt.show()
    print()

def haveTryBPNN():
    sampleinnorm, sampleoutnorm, sampleoutminmax, sampleout = make_data()  # normalizeData1(X_train, y_train) # 数据获取与归一化
    # 训练集，测试集划分
    split_num = int(sampleinnorm.shape[1] * 0.8)
    X_train = sampleinnorm[:, :split_num]
    Y_train = sampleoutnorm[:, :split_num]
    X_test = sampleinnorm[:, split_num:]
    Y_test = sampleoutnorm[:, split_num:]

    bp = BPNN()
    bp.train(X_train, Y_train)
    networkPredict = bp.predict2(X_train)
    # 将预测好的结果缩放到正常的数据范围内；
    networkPredict = scalableData1(networkPredict, sampleoutminmax)
    plotGraph1(sampleout[:, :split_num], bp.errhistory, networkPredict)
    print('Train set evaluation:', end=' ')
    zhibiao = print_evaluate(sampleout[:, :split_num].tolist()[0], networkPredict.tolist()[0])

    # 测试集测试
    textPredict = bp.predict2(X_test)
    textPredict = scalableData1(textPredict, sampleoutminmax)
    plotGraph1(sampleout[:, split_num:], [], textPredict)
    print('Text set evaluation:', end=' ')
    zhibiao1 = print_evaluate(sampleout[:, split_num:].tolist()[0], textPredict.tolist()[0])
    print()
    return zhibiao, zhibiao1

if __name__ == "__main__":
    # haveTry()
    # haveTryBPNN()
    # haveTryBP()

    for i in range(0, 3):
        zhibiao, zhibiao1 = haveTryBPNN()
        zhibiao, zhibiao1 = np.array(zhibiao), np.array(zhibiao1)
    #     if i == 0:
    #         s1, s2 = zhibiao, zhibiao1
    #     else:
    #         s11 = s1 + zhibiao
    #         s22 = s2 + zhibiao1
    #         s1, s2 = s11, s22
    # s1, s2 = s1/3, s2/3
    # print(s1, '\n', s2)


