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

    def __init__(self,file_list,frame_list, transform = None):
        self.framelist = frame_list
        self.filelist = file_list
        self.transform = transform
        self.imagesize1 = 227
        self.labels = [random.randint(0,11) for iter in range(len(self.framelist))]
        self.revs = [random.randint(0,1) for iter in range(len(self.framelist))]
        self.mirrors = [random.randint(0,1) for iter in range(len(self.framelist))]

    def SpatialJit(self,inputimg):
        sjx = self.imagesize1
        sjy = self.imagesize1
        startx = random.randint(0, inputimg.size[0]-sjx)
        starty = random.randint(0, inputimg.size[1]-sjy)
        endx = startx + sjx
        endy = starty + sjy

        sjdis = 5
        sx = random.randint(-sjdis, sjdis)
        sy = random.randint(-sjdis, sjdis)
        if startx + sx > 0 and endx + sx < inputimg.size[0]:
            newx = startx + sx
        else:
            newx = startx    
        if starty + sy > 0 and endy + sy < inputimg.size[1]:
                newy = starty + sy
        else:
            newy = starty 
        imgcrop = inputimg.crop((newx,newy,newx+sjx,newy+sjy))
        return imgcrop

    def __len__(self):
        return len(self.framelist)
    
    def __getitem__(self,idx):

        frame_dir =self.filelist[idx] + '/'
        frames = self.framelist[idx]
        label = self.labels[idx]
        rev = self.revs[idx]
        mirror = self.mirrors[idx]

        fname1 = '/home/jesu2953/data/pulse2/demo1/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[0]).zfill(6))
        fname2 = '/home/jesu2953/data/pulse2/demo1/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[1]).zfill(6))
        fname3 = '/home/jesu2953/data/pulse2/demo1/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[2]).zfill(6))
        fname4 = '/home/jesu2953/data/pulse2/demo1/' + frame_dir + "frames/frame{}_1008x784_576x448.jpg".format(str(frames[3]).zfill(6))

        img1 = Image.open(fname1)
        img2 = Image.open(fname2)
        img3 = Image.open(fname3)
        img4 = Image.open(fname4)

        postproc = [
            transforms.CenterCrop((450,350)),
            transforms.Resize((315,245))
        ]

        self.postproc = transforms.Compose(postproc)

        img1 = self.postproc(img1)
        img2 = self.postproc(img2)
        img3 = self.postproc(img3)
        img4 = self.postproc(img4)

        if mirror == 1:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
            img4 = img4.transpose(Image.FLIP_LEFT_RIGHT)

        imgcrop1 = self.SpatialJit(img1)
        imgcrop2 = self.SpatialJit(img2)
        imgcrop3 = self.SpatialJit(img3)
        imgcrop4 = self.SpatialJit(img4)

        im = []
        im.append(np.expand_dims(np.array(imgcrop1, dtype=np.float32),axis=0))
        im.append(np.expand_dims(np.array(imgcrop2, dtype=np.float32),axis=0))
        im.append(np.expand_dims(np.array(imgcrop3, dtype=np.float32),axis=0))
        im.append(np.expand_dims(np.array(imgcrop4, dtype=np.float32),axis=0))

        ordertype = [[1,2,3,4],[1,3,2,4],[1,3,4,2],[1,2,4,3],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[3,1,2,4],[3,1,4,2],[3,2,1,4]]    
        if rev == 0:
            frame1_idx = im[ordertype[label][0]-1]
            frame2_idx = im[ordertype[label][1]-1]
            frame3_idx = im[ordertype[label][2]-1]
            frame4_idx = im[ordertype[label][3]-1]
        else: 
            frame1_idx = im[ordertype[label][3]-1]
            frame2_idx = im[ordertype[label][2]-1]
            frame3_idx = im[ordertype[label][1]-1]
            frame4_idx = im[ordertype[label][0]-1]

        if self.transform:
            (frame1_idx,frame2_idx,frame3_idx,frame4_idx) = self.transform([frame1_idx,frame2_idx,frame3_idx,frame4_idx])
    

        return frame1_idx,frame2_idx,frame3_idx,frame4_idx,label


class ToTensor(object):
    "change to tensor"

    def __call__(self, list):

        f1 = torch.from_numpy(np.array(list[0]))
        f2 = torch.from_numpy(np.array(list[1]))
        f3 = torch.from_numpy(np.array(list[2]))
        f4 = torch.from_numpy(np.array(list[3]))

        return f1,f2,f3,f4