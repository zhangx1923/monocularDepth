import pandas as pd
from model import MonocularModel
from tools import Visualization

#进行训练
#opt为程序开始时的参数集合，包含以下本函数需要使用的信息：
#random_sample: 是否随机采样，0随机
#in_sample：样本内数据个数
#percent: float,p%为训练集
#df：所有数据（样本内）dataframe
def train(opt, df):
    #划分训练集，测试集
    pass

#df：所有数据（样本外）dataframe
def inference(df):
    pass

    