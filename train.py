from torch._C import dtype
from model import EstDepth_Model, Distance_DS, Detect_Model, Detect_DS
from tools import Visualization
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import train_one_epoch, evaluate
import utils

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
    model = EstDepth_Model(feature_count, 8, 4, 2, label_count)
    if torch.cuda.is_available():
        model.cuda()

    #实例化可视化类
    #vis = Visualization()

    #划分训练集，测试集
    #kf = KFold(opt.fold, shuffle=True, random_state=None)
    kf = KFold(opt.fold)
    for trn_ids, tst_ids in kf.split(df): 
        trn_X, trn_y = df[trn_ids][:, :feature_count], df[trn_ids][:, feature_count:]
        tst_X, tst_y = df[tst_ids][:, :feature_count], df[tst_ids][:, feature_count:]    

        # Generators
        training_set = Distance_DS([ind for ind in range(0, len(trn_y))], trn_X, trn_y)
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
                print(output, local_lables)
            #test
            model.eval()
            output = model(torch.from_numpy(tst_X).float().to(device))
            #计算各种指标
            metrics(output, tst_y)
            print(output, tst_y)
            #可视化

#df：所有数据（样本外）dataframe
def inference(df, feature_count, label_count):
    model = EstDepth_Model(feature_count, 8, 4, 2, label_count)
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict("path")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #实例化可视化类
    vis = Visualization()
    
    for ind in range(len(df)):
        feature, label = df[ind][:feature_count], df[ind][feature_count:]
        output = model(torch.from_numpy(feature).float().to(device))

        #计算各种指标
        metrics(output, label)
        #可视化


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


def testTrainImg(labeldata):

    model = Detect_Model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    batch_size = 4
    epoch_count = 10

    trainset = Detect_DS(label_dir = labeldata, root_dir='dataset/img')
    testset = Detect_DS(label_dir = labeldata, root_dir='dataset/img')

    # split the dataset in train and test set
    indices = torch.randperm(len(trainset)).tolist()
    trainset = torch.utils.data.Subset(trainset, indices[:-50])
    testset = torch.utils.data.Subset(testset, indices[-50:])

    

    trainloader = DataLoader(trainset,\
                             batch_size=batch_size,\
                             shuffle=True,\
                             num_workers=2,\
                             collate_fn=collate_fn)

    
    testloader = DataLoader(testset,\
                            batch_size=1,\
                            shuffle=False,\
                            num_workers=2,\
                            collate_fn=collate_fn)

    classes = ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 
                'Cyclist', 'Tram', 'Misc')
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
   
    for epoch in range(epoch_count):  # loop over the dataset multiple times
        # model.train()
        # for i, (img, bbox, labels, scale_h, scale_w) in tqdm(enumerate(trainloader)):
        #     target = []
        #     img = list(img)
        #     for i in range(len(img)):
        #         d = {}
        #         d["boxes"] = torch.tensor(bbox[i])
        #         d["labels"] = torch.tensor(labels[i],dtype=torch.long)
        #         target.append(d)
        #         img[i] = torch.from_numpy(img[i])
        #     output = model(img, target)

        train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, testloader, device=device)

    print('Finished Training')   
    PATH = 'cifar_net.pth'
    torch.save(model.state_dict(), PATH)

def testTestImg():
    pass
