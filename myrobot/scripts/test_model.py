from unittest import result
import torch
import cv2
import torchvision
import torchvision.transforms as transforms
from model import Net
class Inference():
    def __init__(self,dir="src/myrobot/scripts/cifar_net.pth"):

        self.net=Net()
        self.net.load_state_dict(torch.load(dir))
        self.classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        #preprocess img
    def preprocess(self,img):
        transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img_re=cv2.resize(img,(32,32))

        return transform(img_re)

    def run(self,img):
        img=self.preprocess(img)
        results=self.net(img.unsqueeze(0))
        print(results)
        _,pred=torch.max(results,1)

        print(self.classes[pred])
        return self.classes[pred]

# if __name__ == '__main__':
#     model=Inference()
#     img=cv2.imread("/home/cuong/Desktop/ROS/src/myrobot/scripts/truck.jpg")
#     model.run(img)
