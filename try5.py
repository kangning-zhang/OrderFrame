import time
import os
import glob
import json
import torch
import random
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class PulseDataset(Dataset):
    "Loading PulseDataset"

    def __init__(self,file_list,frame_list,transform = None):
        self.framelist = frame_list
        self.filelist = file_list
        self.transform = transform
        self.frame1 = []
        self.frame2 = []
        self.frame3 = []
        self.frame4 = []
        self.label = []

        for cnt in range(len(self.framelist)):

            frame_dir =self.filelist[cnt] + '/'
            frames = self.framelist[cnt]

            fname1 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[0]).zfill(6))
            fname2 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[1]).zfill(6))
            fname3 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[2]).zfill(6))
            fname4 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[3]).zfill(6))

            img1 = Image.open(fname1)
            img2 = Image.open(fname2)
            img3 = Image.open(fname3)
            img4 = Image.open(fname4)
                            
            im = [[None]*1] * 4
            im[0][0] = np.array(img1).transpose((1,0))
            im[1][0] = np.array(img1).transpose((1,0))
            im[2][0] = np.array(img1).transpose((1,0))
            im[3][0] = np.array(img1).transpose((1,0))

            order = random.randint(0,11)
            rev = random.randint(0,1)
            ordertype = [[1,2,3,4],[1,3,2,4],[1,3,4,2],[1,2,4,3],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[3,1,2,4],[3,1,4,2],[3,2,1,4]]    
            if rev == 0:
                self.frame1.append(im[ordertype[order][0]-1])
                self.frame2.append(im[ordertype[order][1]-1])
                self.frame3.append(im[ordertype[order][2]-1])
                self.frame4.append(im[ordertype[order][3]-1])
            else: 
                self.frame1.append(im[ordertype[order][3]-1])
                self.frame2.append(im[ordertype[order][2]-1])
                self.frame3.append(im[ordertype[order][1]-1])
                self.frame4.append(im[ordertype[order][0]-1])
                
            self.label.append(order)


    def __len__(self):
        return len(self.framelist)
    
    def __getitem__(self,idx):

        #frame1_idx = torch.from_numpy(np.array(self.frame1)[idx])
        #frame2_idx = torch.from_numpy(np.array(self.frame2)[idx])
        #frame3_idx = torch.from_numpy(np.array(self.frame3)[idx])
        #frame4_idx = torch.from_numpy(np.array(self.frame4)[idx])
        #label_idx = torch.as_tensor(np.array(self.label)[idx])

        frame1_idx = np.array(self.frame1)[idx]
        frame2_idx = np.array(self.frame2)[idx]
        frame3_idx = np.array(self.frame3)[idx]
        frame4_idx = np.array(self.frame4)[idx]
        label_idx = np.array(self.label)[idx]


        if self.transform:
            (frame1_idx,frame2_idx,frame3_idx,frame4_idx,label_idx) = self.transform([frame1_idx,frame2_idx,frame3_idx,frame4_idx,label_idx])

        return frame1_idx,frame2_idx,frame3_idx,frame4_idx,label_idx


class ToTensor(object):
    "change to tensor"

    def __call__(self, list):

        f1 = torch.from_numpy(list[0])
        f2 = torch.from_numpy(list[1])
        f3 = torch.from_numpy(list[2])
        f4 = torch.from_numpy(list[3])
        l = torch.as_tensor(list[4])

        return f1,f2,f3,f4,l


### Loading framelist and filelist for both train&val set ###

def loadList(filename):
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

train_file = Path(".").resolve() / "SPD_train1.json"
with open(train_file, 'r') as f:
    train_scan_id = json.load(f)

test_file = Path(".").resolve() / "SPD_test1.json"
with open(test_file, 'r') as f:
    test_scan_id = json.load(f)

train_frame = loadList('trainframe.npy') # frames choosen, shape = (#tuple * 4)
train_keep = loadList('trainkeep.npy')  # position of scan id in train_scan_id corresponding to each frame choosen
test_frame = loadList('testframe.npy')
test_keep = loadList('testkeep.npy')

train_file = [None] * len(train_keep)
test_file = [None] * len(test_keep)
for i in range(len(train_keep)):
    train_file[i] = train_scan_id[train_keep[i]]  # scan id corresponding to each frame choosen
for i in range(len(test_keep)):
    test_file[i] = test_scan_id[test_keep[i]]  # scan id corresponding to each frame choosen



traindata = PulseDataset(train_file[0:201], train_frame[0:201], transform=transforms.Compose([ToTensor()]))
# having 28597 tumples as training dataset, but if I input all the tumples, it get killed in the for loop below
# with error msg RuntimeError: DataLoader worker (pid 14270) is killed by signal: Killed.
# If I just input for example the first 200 tuples (smaller size), it works
trainloader = torch.utils.data.DataLoader(traindata, batch_size=32,num_workers = 1)

print('start iteration:')


for i, (frame1, frame2, frame3, frame4, label) in enumerate(trainloader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (i%5 == 0):

        'print out information every 5 iteration'

        input1 = frame1.to(device)
        input2 = frame2.to(device)
        input3 = frame3.to(device)
        input4 = frame4.to(device)
        label = label.to(device)

        print(i)
        print(label.shape)
        print(input1.shape)






