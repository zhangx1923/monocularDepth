import pandas as pd
from model import EstDepth, Dataset
from tools import Visualization
from sklearn.model_selection import KFold
import torch

def metrics(pred: list, label: list):
    pass

#进行训练
#opt为程序开始时的参数集合，包含以下本函数需要使用的信息：
#random_sample: 是否随机采样，0随机
#in_sample：样本内数据个数
#df：所有数据（样本内）dataframe
#feature_count数据集中特征的个数
#label_count数据集中标签的个数
def train(opt, df, feature_count, label_count):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MonocularModel()
    if torch.cuda.is_available():
        model.cuda()

    #实例化可视化类
    vis = Visualization()

    #划分训练集，测试集
    #kf = KFold(opt.fold, shuffle=True, random_state=None)
    kf = KFold(opt.fold)
    for trn_ids, tst_ids in kf.split(df): 
        trn_X, trn_y = df[trn_ids][:, :feature_count], df[trn_ids][:, feature_count:]
        tst_X, tst_y = df[tst_ids][:, :feature_count], df[tst_ids][:, feature_count:]    

        # Generators
        training_set = Dataset([ind for ind in range(0, len(trn_y))], trn_X, trn_y)
        training_generator = torch.utils.data.DataLoader(training_set, opt.batch, shuffle=True)
        
        #train + test
        for ep in range(opt.epoch):
            #train
            model.train()
            for ii, (local_batch, local_lables) in enumerate(training_generator):
                local_batch, local_lables = local_batch.to(device), local_lables.to(device)
                output = model(local_batch)
                #计算各种指标
                metrics(output, local_lables)
                #可视化
            #test
            model.eval()
            output = model(tst_X)
            #计算各种指标
            metrics(output, tst_y)
            #可视化

#df：所有数据（样本外）dataframe
def inference(df, feature_count, label_count):
    model = MonocularModel()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict()
    model.eval()

    #实例化可视化类
    vis = Visualization()
    
    for ind in range(len(df)):
        feature, label = df[ind][:feature_count], df[ind][feature_count:]
        output = model(feature)

        #计算各种指标
        metrics(output, label)
        #可视化

    