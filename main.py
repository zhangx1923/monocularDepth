from torchvision import models, transforms
import torch
from PIL import Image

#Image Classification 
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
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        
        #load file containing top 1000 labels for the ImageNet dateset
        with open("top1000labels.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        
        result = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

        print(result)


#Semantic Segmentation
def test_fcn():
    #load pretrained model
    resnet = models.resnet101(pretrained=True)
    #set eval mode
    resnet.eval()

#obejct detection
def test_fasterrcnn():
    pass

if __name__ == "__main__":
    test_resnet()