from torch._C import dtype
from model import EstDepth_Model, Distance_DS, Detect_Model, Detect_DS
from tools import Visualization
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
import utils
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt

def metrics(pred, label):
    #pred = pd.DataFrame(pred.cpu().detach().numpy()).fillna(0)
    rmse_dnn = sqrt(mean_squared_error(label, pred))
    r2_dnn = r2_score(label, pred)  
    mae_dnn = mean_absolute_error(label, pred)
    return rmse_dnn, r2_dnn, mae_dnn

#进行训练
#opt为程序开始时的参数集合，包含以下本函数需要使用的信息：
#random_sample: 是否随机采样，0随机
#in_sample：样本内数据个数
#df：所有数据（样本内）dataframe
#feature_count数据集中特征的个数
#label_count数据集中标签的个数
def trainDecoder(opt, df, feature_count, label_count, log):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EstDepth_Model(feature_count, 8, 4, 2, label_count)
    if torch.cuda.is_available():
        model.cuda()

    #实例化可视化类
    #vis = Visualization()

    #划分训练集，测试集
    #kf = KFold(opt.fold, shuffle=True, random_state=None)
    kf = KFold(opt.fold)
    current_k = 1

    best_model_name = ""

    for trn_ids, tst_ids in kf.split(df): 
        trn_X, trn_y = df[trn_ids][:, :feature_count], df[trn_ids][:, feature_count:]
        tst_X, tst_y = df[tst_ids][:, :feature_count], df[tst_ids][:, feature_count:]    

        # Generators
        training_set = Distance_DS([ind for ind in range(0, len(trn_y))], trn_X, trn_y)
        training_generator = torch.utils.data.DataLoader(training_set, opt.batch, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr = opt.lr)

        rmse_list, r2_list, mae_list = [], [], []

        #train + test
        for ep in range(opt.epoch):
            #train
            model.train()
            for ii, (local_batch, local_lables) in enumerate(training_generator):
                local_batch, local_lables = local_batch.to(device), local_lables.to(device)
                output = model(local_batch)
                # print(output, local_lables)
                loss = criterion(output, local_lables)     
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #test
            model.eval()
            output = model(torch.from_numpy(tst_X).float().to(device))
            #计算各种指标
            rmse, r2, mae = metrics(pd.DataFrame(output.cpu().detach().numpy()).fillna(0), tst_y)
            rmse_list.append(rmse)
            r2_list.append(r2)
            mae_list.append(mae)
            log.print("K-fold:{}, epoch:{}, r2:{}, rmse:{}, mae:{}".format(current_k, ep+1, r2, rmse, mae))
            #可视化
            #????
        xaris = [i for i in range(1,opt.epoch+1)]
        plt.subplot(2,2,1)
        plt.plot(xaris, r2_list)
        plt.xlabel("epoch")
        plt.ylabel("R2")

        plt.subplot(2,2,2)
        plt.plot(xaris, rmse_list)
        plt.xlabel("epoch")
        plt.ylabel("rmse")

        plt.subplot(2,2,3)
        plt.plot(xaris, mae_list)
        plt.xlabel("epoch")
        plt.ylabel("mae")

        k_file_name = log.file_location + "/fold-" + str(current_k)

        plt.savefig(k_file_name + ".jpg")
        plt.close(0)
        
        torch.save(model.state_dict(), k_file_name + ".pkl")

        #选择最好的model返回其路径，这里暂时只返回最后一个flod的
        best_model_name = k_file_name + ".pkl"
        current_k += 1
    return best_model_name
        

#df：所有数据（样本外）dataframe
def inferenceDecoder(df, feature_count, label_count, log, model_name):
    model = EstDepth_Model(feature_count, 8, 4, 2, label_count)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # #实例化可视化类
    # vis = Visualization()
    
    output_list, label_list = [], []
    for ind in range(len(df)):
        feature, label = df[ind][:feature_count], df[ind][feature_count:]
        output = model(torch.from_numpy(feature).float().to(device)).cpu().detach().numpy()  

        output_list.append(output)
        label_list.append(label)
        log.print("ind:{}, inference:{}, label:{}".format(ind, output, label))
    #计算各种指标
    rmse, r2, mae = metrics(output_list, label_list)
    log.print("r2:{}, rmse:{}, mae:{}".format(r2, rmse, mae))
        #可视化
        #???
        


#augment dataload to solve one image with multi obejcts, so that tensor size is different
def collate_fn(batch):
    return tuple(zip(*batch))
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
    """

    # images = list()
    # boxes = list()
    # labels = list()
    # scale_h = list()
    # scale_w =  list()

    # for b in batch:
    #     images.append(b[0])
    #     boxes.append(b[1])
    #     labels.append(b[2])
    #     scale_h.append(b[3])
    #     scale_w.append(b[4])
        
    # images = torch.stack(torch.from_numpy(images), dim=0)
    # scale_h = torch.stack(torch.from_numpy(scale_h), dim=0)
    # scale_w = torch.stack(torch.from_numpy(scale_w), dim=0)

    # return images, boxes, labels , scale_h, scale_w


def trainEncoder(labeldata, opt, log):
    model = Detect_Model(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)

    trainset = Detect_DS(label_dir = labeldata, root_dir='dataset/img')
    testset = Detect_DS(label_dir = labeldata, root_dir='dataset/img')

    indices = torch.randperm(len(trainset)).tolist()

    trainset = torch.utils.data.Subset(trainset, indices[:opt.ts])
    testset = torch.utils.data.Subset(testset, indices[opt.ts:5])

    

    trainloader = DataLoader(trainset,\
                            batch_size=opt.batch,\
                            shuffle=True,\
                            num_workers=opt.nw,\
                            collate_fn=collate_fn)

    
    testloader = DataLoader(testset,\
                            batch_size=opt.batch,\
                            shuffle=False,\
                            num_workers=opt.nw,\
                            collate_fn=collate_fn)

    classes = ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 
                'Cyclist', 'Tram', 'Misc')

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(opt.epoch):  # loop over the dataset multiple times
        train_one_epoch(model, optimizer, trainloader, device, epoch, opt.print_freq, log)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, testloader, log, device=device)

    log.print('Finished Training!')   
    PATH = log.file_location + '/encoder.pth'
    torch.save(model.state_dict(), PATH)
    return PATH

# def inferenceEncoder(model_path):
#     model = Detect_Model(False)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     # device = torch.device('cpu')
#     model.to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
