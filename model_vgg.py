import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderNetwork(nn.Module):
    def __init__(self):
        super(OrderNetwork, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(7*7*512,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096*2,4096,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096*6,12,bias=True),
        )

    def forward_once(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        f6 = self.fc1(x)
        return f6
    
    def forward(self, input1, input2, input3 ,input4, batchsize = 128):
        inputall = torch.cat((input1,input2,input3,input4),0)
        f6 = self.forward_once(inputall)
        f6_1 = f6[0:batchsize]
        f6_2 = f6[batchsize:batchsize*2]
        f6_3 = f6[batchsize*2:batchsize*3]
        f6_4 = f6[batchsize*3:batchsize*4]
        f7_1 = self.fc2(torch.cat((f6_1,f6_2),1))
        f7_2 = self.fc2(torch.cat((f6_1,f6_3),1))
        f7_3 = self.fc2(torch.cat((f6_1,f6_4),1))
        f7_4 = self.fc2(torch.cat((f6_2,f6_3),1))
        f7_5 = self.fc2(torch.cat((f6_2,f6_4),1))
        f7_6 = self.fc2(torch.cat((f6_3,f6_4),1))
        f7_all = torch.cat((f7_1,f7_2,f7_3,f7_4,f7_5,f7_6),1)
        output = self.classifier(f7_all)
        return output


net = OrderNetwork()