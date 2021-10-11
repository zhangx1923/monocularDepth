import argparse
from data import GenerateData
from tools import Log, Conf
from train import train, inference
import os
import time
import pandas as pd
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #执行相关参数
    parser.add_argument('--result_path', help="file location, including picture and saved model, such as xxx", required=False, type=str, default= "-1")
    #parser.add_argument('--img_path', help="存放用于训练和测试的image路径",required=False, type=str, default="img")
    parser.add_argument('--mode', help="predict用于直接用现有模型去预测，train用于重新训练模型，ctrain用于在现有模型上继续训练",required=False, type=str, default="train")
    
    #数据集参数
    parser.add_argument('--generate_data', help="是否重新生成数据,new--重新生成，old--旧数据", required=False, default="old", type=str)
    parser.add_argument('--in_sample', help='the number of train set and val set', required=False, type=int, default=70000)
    parser.add_argument('--random_sample', help="是否随机采样,1随机", required=False, default=1, type=int)
    parser.add_argument('--percent', help="用于train的数据占in sample 的比例",required=False, type=float, default=.7)
    
    #模型参数
    parser.add_argument('--fold', help='k-fold cross validation', required=False, type=int, default=5)

    opt = parser.parse_args()
    
    #建立存放结果的文件夹和初始化写文件类
    loc, i = opt.result_path if opt.result_path != "-1" else str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())), 1
    #检查该文件夹是否存在
    while os.path.exists(loc):
        loc += i
    os.mkdir(loc)
    log = Log(loc)

    #获取配置文件信息
    conf = Conf()
    CSVdata = conf.getContent("csv_data")

    #获取数据
    if opt.generate_data != "old" or not os.path.exists(CSVdata):
        GenerateData(conf.getContent("img_data"))
    df = pd.read_csv(CSVdata)

    if opt.in_sample >= len(df):
        log.error("样本内数据大于总数据数量，请重新指定in_sample值!")

    #划分样本内数据（训练+验证），样本外数据（测试集）
    if opt.random_sample != 0:
        #随机采样(无放回)
        index = [i for i in range(0, len(df))]
        random.shuffle(index)
        in_sample, out_sample = df.iloc[index[:opt.in_sample],:-1].values, df.iloc[index[opt.in_sample:],:-1].values
    else:
        print("无随机")
        in_sample, out_sample= df.iloc[:opt.in_sample,:-1].values, df.iloc[opt.in_sample:,:-1].values

    train(opt, in_sample)

    inference(out_sample)
