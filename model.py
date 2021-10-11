from torchvision import models, transforms
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
import torch
from PIL import Image
from torchvision.transforms.functional import normalize
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes

class MonocularModel():
    pass


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
    for i in range(2):
        img_name = str(i)
        while len(img_name) < 6:
            img_name = "0" + img_name
        #img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))
        img_list.append(transform(read_image(img_path + img_name + ".png").to(device)))

    batch_int = torch.stack(img_list)
    batch = convert_image_dtype(batch_int, dtype=torch.float)

    model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
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
#     test_mask_rcnn()