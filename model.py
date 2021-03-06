from torch.nn.modules.linear import Linear
from torchvision import models, transforms, ops
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
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
from torchvision.models.detection.rpn import AnchorGenerator

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

#??????????????????????????????????????????
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
        #?????????????????????????????????????????????   ??????????????????????????????
        # img, params = util.random_flip(
        #     img, x_random=True, return_param=True)
        # bbox = util.flip_bbox(
        #     bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scaleH, scaleW

#????????????????????????????????????
#??????????????????object?????????label?????????????????????agument dataloader's collate_fn
#dataset ????????????id?????????????????????????????????obejct bbox???
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


#object detect model
#pre_train_para means wherther use pretrained parameter, boolean
def Detect_Model(pre_train_para):
    model = fasterrcnn_resnet50_fpn(pretrained=pre_train_para)
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # replace the classifier with a new one, that has
    # num_classes which is user-defined

    #???????????????kitti???coco??????????????????
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

#pre_train_para  stands for whether use pretrained model
#freeze stands for whether freeze the feature part weights and bias
def Detect_Model1(pre_train_para, freeze):
    # load a pre-trained model for classification and return
    # only the features
    backbone = models.mobilenet_v2(pretrained=pre_train_para).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                    num_classes=4,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    # print(list(model.parameters())[-5:])

    # for i,p in enumerate(model.parameters()):
    #     print(i, p)
    if freeze:
        for i, p in enumerate(model.parameters()):
            #?????????????????????           
            if i < 166:
                p.requires_grad = False

    return model

#input 4??????, xmin, ymin, xmax, ymax,
#output 1???, zloc=distance
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

    #int?????????????????????????????????
    batch_int = torch.stack(img_list)

    #????????????????????????float??????
    batch = convert_image_dtype(torch.stack(img_list), dtype=torch.float)
    
    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # print(output.shape, output.min().item(), output.max().item())
    normalized_masks = torch.nn.functional.softmax(output, dim=1)

    # #?????????????????????????????????????????????,???????????????????????????????????????
    # car_person_bicycle_bus_motorbike_masks = [
    #     normalized_masks[img_idx, sem_class_to_idx[cls]]
    #     for img_idx in range(batch.shape[0])
    #     for cls in ('car','person','bicycle','bus','motorbike')
    # ]
    # show(car_person_bicycle_bus_motorbike_masks)

    #?????????????????????????????????torchvision.utils.draw_segmentation_masks
    #????????????????????????boolean masks,????????????????????????
    #???????????????class
    # class_dim = 1
    # boolean_person_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx['person'])
    # #show([m.float() for m in boolean_person_masks])
    # person_with_masks = [
    #     draw_segmentation_masks(img_batch, masks=mask, alpha=0.9, colors='red')
    #     for img_batch, mask in zip(batch_int.to('cpu'), boolean_person_masks)
    # ]
    # show(person_with_masks)

    #?????????class
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
    #???????????????????????????0.8???obejct
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
    proba_threshold = 0.5  #??????????????????
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

