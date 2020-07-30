import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrderNetwork(nn.Module):
    def __init__(self):
        super(OrderNetwork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2,groups = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(3, 2),
            nn.LocalResponseNorm(5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace = True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1,groups = 2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace = True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1,groups = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(3, 2),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(6*6*256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(1024*2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p=0.5)
        )       
        self.fc8 = nn.Sequential(
            nn.Linear(512*6, 12)
        )

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x,1)
        f6 = self.fc6(x)
        return f6

    
    def forward(self, input1, input2, input3 ,input4, batchsize = 128):
        inputall = torch.cat((input1,input2,input3,input4),0)
        f6 = self.forward_once(inputall)
        f6_1 = f6[0:batchsize]
        f6_2 = f6[batchsize:batchsize*2]
        f6_3 = f6[batchsize*2:batchsize*3]
        f6_4 = f6[batchsize*3:batchsize*4]
        f7_1 = self.fc7(torch.cat((f6_1,f6_2),1))
        f7_2 = self.fc7(torch.cat((f6_1,f6_3),1))
        f7_3 = self.fc7(torch.cat((f6_1,f6_4),1))
        f7_4 = self.fc7(torch.cat((f6_2,f6_3),1))
        f7_5 = self.fc7(torch.cat((f6_2,f6_4),1))
        f7_6 = self.fc7(torch.cat((f6_3,f6_4),1))
        f7_all = torch.cat((f7_1,f7_2,f7_3,f7_4,f7_5,f7_6),1)
        output = self.fc8(f7_all)
        return output
        

net = OrderNetwork()