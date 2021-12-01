from torchvision import models, transforms
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from PIL import Image
from torchvision.transforms.functional import normalize
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import pandas as pd
from data import GenerateData
import transforms as T
import util
from skimage import transform as sktsf

def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess(img, height, weight, scale):
    """Preprocess an image for feature extraction.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    # scaleH = H / height
    # scaleW = W / weight
    img = img / 255.
    img = sktsf.resize(img, (C, height * scale, weight*scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    return pytorch_normalze(img)

#用于训练获取深度信息的数据集
class Distance_DS(torch.utils.data.Dataset):
  def __init__(self, list_IDs, attr, label):
        self.attr = attr
        self.label = label
        self.list_IDs = list_IDs

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X, y = self.attr[ID], self.label[ID] 
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

class Transform(object):

    def __init__(self, height=375, weight=1242, scale=.8):
        self.height = height
        self.weight = weight
        self.scale = scale

    def __call__(self, in_data):
        img, bbox, label = in_data
        # print(img, bbox, label)
        _, H, W = img.shape
        img = preprocess(img, self.height, self.weight, self.scale)
        _, o_H, o_W = img.shape
        scaleH = o_H / H
        scaleW = o_W / W
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        #horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scaleH, scaleW

#用于获取目标检测的数据集
#一张图片多个object，导致label的维度不一致，agument dataloader's collate_fn
#dataset 要求一个id对应一张图片，而非一个obejct bbox、
class Detect_DS(Dataset):
    def __init__(self, label_dir, root_dir):
        """
        Args:
            label_dir (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_dir = label_dir
        self.root_dir = root_dir
        self.transform = Transform()

    def __len__(self):
        label_files, img_files = os.listdir(self.label_dir), os.listdir(self.root_dir)
        assert len(label_files) == len(img_files)
        return len(label_files)

    #idx stands for image id (image name = "00..0"+idx + ".jpg")
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(idx)
        while len(img_name) < 6:
            img_name = "0" + img_name
        label_name = img_name
        img_name += ".png"
        label_name += ".txt"
        img_path = os.path.join(self.root_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image, dtype=np.float32)

        gd = GenerateData()
        type_to_int = gd.get_type_int()

        with open(label_path) as f:
            label_region = list(l.split("\n")[0].split(" ") for l in f.readlines() if "DontCare" not in l)
        #print(label_region)
        labels, bbox = list(), list()
        # print(label_region)
        for l in label_region:
            if l[0] not in type_to_int.keys():
                continue
            labels.append(type_to_int[l[0]])
            bbox.append([l[i] for i in [4,5,6,7]])

        bbox = np.stack(bbox).astype(np.float32)
        labels = np.stack(labels).astype(np.int64)
        # print(bbox, labels)
        if image.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            image = image[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))

        image, bbox, labels, scaleH, scaleW = self.transform((image, bbox, labels))
        
        target = {}
        target["boxes"] = torch.from_numpy(bbox.copy())
        target["labels"] = torch.from_numpy(labels.copy())
        #target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        area = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["area"] = area
        # suppose all instances are not crowd
        num_objs = len(labels)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target["iscrowd"] = iscrowd


        return torch.from_numpy(image.copy()), target.copy()
        #print(image.copy().shape, bbox.copy().shape, labels.copy().shape)
        #print(target)
        #print(bbox.shape, image.shape)
        #return image.copy(), bbox.copy(), labels.copy(), [scaleH], [scaleW]



# class Detect_DS(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             label_dir (string): Path to the txt file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.label = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.label)

#     #idx stands for image id (image name = "00..0"+idx + ".jpg")
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.label.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB").resize((1200,370))

#         label_region_useful = self.label.iloc[idx, [1,5,6,7,8]]
        
#         #print(label_region_useful[0])
#         targets = {}
#         targets["label"] = label_region_useful[0]
#         targets["boxes"] = [label_region_useful[1],label_region_useful[2],label_region_useful[3],label_region_useful[4]]
#         # targets = np.array([targets])
#         # label_region_useful = label_region_useful.astype('float').reshape(-1, 2)
#         if self.transform is not None:
#             image = self.transform(image)

#         return image, targets

#object detect model
#pre_train_para means wherther use pretrained parameter, boolean
def Detect_Model(pre_train_para):
    model = fasterrcnn_resnet50_fpn(pretrained=pre_train_para)
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # replace the classifier with a new one, that has
    # num_classes which is user-defined

    #目前只检测kitti和coco重合的部分：
    # car  cocotype 3 kitti 0
    # Pedestrian coco 1 kitti 3
    # bicycle coco 2 kitti 5
    # truck  coco 8 kitti 2
    num_classes = 4  # 1 class (person) + background
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# class Detect_Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

#input 4维度, xmin, ymin, xmax, ymax,
#output 1维, zloc=distance
class EstDepth_Model(nn.Module):
    def __init__(self, in_dim, n_hidden_1,n_hidden_2,n_hidden_3, out_dim):
        super(EstDepth_Model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


plt.rcParams["savefig.bbox"] = 'tight'
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("test.jpg")

#Image Classification 
#output: probablity of top 5 possible classifications
def test_resnet():
    #load pretrained model
    resnet = models.resnet101(pretrained=True)
    #set eval mode
    resnet.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    #test picture path
    img_path = "test img/"

    for i in range(1):
        img_name = str(i)
        while len(img_name) < 6:
            img_name = "0" + img_name
        img = transform(Image.open(img_path + img_name + ".png"))
        batch_t = torch.unsqueeze(img, 0)
        out = resnet(batch_t)
        #analyze results
        _, indices = torch.sort(out, descending=True)
        percentage = F.softmax(out, dim=1)[0] * 100
        
        #load file containing top 1000 labels for the ImageNet dateset
        with open("top1000labels.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        
        result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

        print(result)

#Semantic Segmentation
def test_fcn():
    #load pretrained model
    model = fcn_resnet50(pretrained=True, progress=False)
    if torch.cuda.is_available():
        model.cuda()
    #set eval mode
    model.eval()
    #test picture path
    img_path = "test img/"
    
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    
    img_list = []
    transform = transforms.Compose([
        transforms.Resize((37*5,122*5))
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(2):
        img_name = str(i)
        while len(img_name) < 6:
            img_name = "0" + img_name
        #img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))
        img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))

    #int类型，用于在原图上绘制
    batch_int = torch.stack(img_list)

    #用于画小图，需要float类型
    batch = convert_image_dtype(torch.stack(img_list), dtype=torch.float)
    
    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # print(output.shape, output.min().item(), output.max().item())
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # #按照选定的分类，每个分类画一个,黑底，检测出来的为其他颜色
    # car_person_bicycle_bus_motorbike_masks = [
    #     normalized_masks[img_idx, sem_class_to_idx[cls]]
    #     for img_idx in range(batch.shape[0])
    #     for cls in ('car','person','bicycle','bus','motorbike')
    # ]
    # show(car_person_bicycle_bus_motorbike_masks)

    #如果想在原图上画，使用torchvision.utils.draw_segmentation_masks
    #这个函数要求得到boolean masks,使用如下方法得到
    #只能画一个class
    # class_dim = 1
    # boolean_person_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
    # #show([m.float() for m in boolean_person_masks])
    # person_with_masks = [
    #     draw_segmentation_masks(img_batch, masks=mask, alpha=0.9, colors='red')
    #     for img_batch, mask in zip(batch_int.to('cpu'), boolean_person_masks)
    # ]
    # show(person_with_masks)

    #画多个class
    class_dim = 1
    num_classes = normalized_masks.to('cpu').shape[1]
    all_classes_masks = normalized_masks.to('cpu').argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
    # The first dimension is the classes now, so we need to swap it
    all_classes_masks = all_classes_masks.swapaxes(0, 1)

    img_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=0.9)
        for img, mask in zip(batch_int.to('cpu'), all_classes_masks)
    ]
    show(img_with_masks)
        

#obejct detection
def test_fasterrcnn():
    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    #test picture path
    img_path = "test img/"

    img_list = []
    transform = transforms.Compose([
        transforms.Resize((37*5,122*5))
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(60,61):
        img_name = str(i)
        while len(img_name) < 6:
            img_name = "0" + img_name
        #img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))
        img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))

    batch_int = torch.stack(img_list)
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
    # num_classes = 91  # 1 class (person) + background
    # # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if torch.cuda.is_available():
        model.cuda()
    model = model.eval()
    
    outputs = model(batch)
    print(outputs)
    #绘制出来可能性大于0.8的obejct
    score_threshold = .8
    img_with_boxes = [
        draw_bounding_boxes(img_int.cpu(), boxes=output['boxes'][output['scores'] > score_threshold].cpu(), width=4)
        for img_int, output in zip(batch_int, outputs)
    ]
    show(img_with_boxes)

#instance segmentation
def test_mask_rcnn():
    #test picture path
    img_path = "test img/"

    img_list = []
    transform = transforms.Compose([
        transforms.Resize((37*3,122*3))
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(2):
        img_name = str(i)
        while len(img_name) < 6:
            img_name = "0" + img_name
        #img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))
        img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))

    batch_int = torch.stack(img_list)
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
    if torch.cuda.is_available():
        model.cuda()
    model = model.eval()

    output = model(batch)
    #print(output)

    inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

    print([inst_classes[label] for label in output[0]['labels']])
    print(output[0]['scores'])
    proba_threshold = 0.5  #这是干嘛的？
    score_threshold = .75

    boolean_masks = [
        out['masks'][out['scores'] > score_threshold] > proba_threshold
        for out in output
    ]

    img_with_masks = [
        draw_segmentation_masks(img, mask.squeeze(1),alpha=.9, colors='red')
        for img, mask in zip(batch_int.to('cpu'), boolean_masks)
    ]
    show(img_with_masks)

# if __name__ == "__main__":
#     test_fasterrcnn()

