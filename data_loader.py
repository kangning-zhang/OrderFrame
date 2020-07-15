from random import shuffle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import time
import random
import torch

class train_loader:
    def __init__(self,config):
        self.imagesize1 = config.imagesize1
        self.imagesize2 = config.imagesize2

    def load(self):

        top = [None] * 5

        def loadList(filename):
            tempNumpyArray=np.load(filename)
            return tempNumpyArray.tolist()

        file_list = loadList('scan.npy')

        self.tuplenum = len(file_list)
        self.batchsize = len(file_list)
        self.filelist = file_list     
        self.randlist = list(range(self.tuplenum))
        shuffle(self.randlist)
        self.idxcounter = 0
        self.framelist = [1997,2000,2003,2006]

        self.channel = 1
        self.height = self.imagesize1
        self.width = self.imagesize2

        top_names = ['im1','im2','im3','im4','label']
        for top_index, name in enumerate(top_names):
            if name == 'label':
                shape = (self.batchsize,)
            else:
                shape = (self.batchsize, self.height,self.width)
        self.im1 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        self.im2 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        self.im3 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        self.im4 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        self.label = torch.zeros((self.batchsize,))


        tmpdata1 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        tmpdata2 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        tmpdata3 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        tmpdata4 = torch.zeros((self.batchsize, self.channel, self.height, self.width))
        for cnt in range(0,self.batchsize):

            frame_dir = self.filelist[ self.randlist[self.idxcounter] ] + '/'
            
            mirror = random.randint(0,1)

            fname1 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame%06d_1008x784_576x448.jpg"%(self.framelist[0])
            fname2 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame%06d_1008x784_576x448.jpg"%(self.framelist[1])
            fname3 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame%06d_1008x784_576x448.jpg"%(self.framelist[2])
            fname4 = '/home/jesu2953/data/pulse2/demo/' + frame_dir + "frames/frame%06d_1008x784_576x448.jpg"%(self.framelist[3])

            img1 = Image.open(fname1)
            img2 = Image.open(fname2)
            img3 = Image.open(fname3)
            img4 = Image.open(fname4)
            
            ## Mirror
            if mirror == 1:
                img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
                img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)
                img4 = img4.transpose(Image.FLIP_LEFT_RIGHT)

            ## Spatial Jittering
            sjx = self.imagesize1
            sjy = self.imagesize2
            startx = random.randint(0, img1.size[0]-sjx)
            starty = random.randint(0, img1.size[1]-sjy)
            endx = startx + sjx
            endy = starty + sjy

            sjdis = 5
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop1 = img1.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop2 = img2.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop3 = img3.crop((newx,newy,newx+sjx,newy+sjy))
            sx = random.randint(-sjdis, sjdis)
            sy = random.randint(-sjdis, sjdis)
            if startx + sx > 0 and endx + sx < img1.size[0]:
                newx = startx + sx
            else:
                newx = startx    
            if starty + sy > 0 and endy + sy < img1.size[1]:
                newy = starty + sy
            else:
                newy = starty 
            imgcrop4 = img4.crop((newx,newy,newx+sjx,newy+sjy))
        
            im1 = torch.from_numpy(np.array(imgcrop1, dtype=np.float32).transpose((1,0)))
            im2 = torch.from_numpy(np.array(imgcrop2, dtype=np.float32).transpose((1,0)))
            im3 = torch.from_numpy(np.array(imgcrop3, dtype=np.float32).transpose((1,0)))
            im4 = torch.from_numpy(np.array(imgcrop4, dtype=np.float32).transpose((1,0)))
            

            ''' Ultrasound Image is grey, no RGB channel    
            ## Channel Splitting
            rgb = random.randint(0,2)
            im1 = im1[:,:,rgb]
            rgb = random.randint(0,2)
            im2 = im2[:,:,rgb]
            rgb = random.randint(0,2)
            im3 = im3[:,:,rgb]
            rgb = random.randint(0,2)
            im4 = im4[:,:,rgb]
            im1 = np.stack((im1,)*3, axis=2)
            im2 = np.stack((im2,)*3, axis=2)
            im3 = np.stack((im3,)*3, axis=2)
            im4 = np.stack((im4,)*3, axis=2)
            im1 -= 96.5 
            im2 -= 96.5 
            im3 -= 96.5 
            im4 -= 96.5
                
            im1 = im1[:,:,::-1]
            im2 = im2[:,:,::-1]
            im3 = im3[:,:,::-1]
            im4 = im4[:,:,::-1]
            '''

            im = [None] * 4
            im[0] = im1
            im[1] = im2
            im[2] = im3
            im[3] = im4
            
            order = random.randint(0,11)
            rev = random.randint(0,1)
            ordertype = [[1,2,3,4],[1,3,2,4],[1,3,4,2],[1,2,4,3],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[3,1,2,4],[3,1,4,2],[3,2,1,4]]    
            if rev == 0:
                tmpdata1[cnt][0][:] = im[ordertype[order][0]-1]
                tmpdata2[cnt][0][:] = im[ordertype[order][1]-1]
                tmpdata3[cnt][0][:] = im[ordertype[order][2]-1]
                tmpdata4[cnt][0][:] = im[ordertype[order][3]-1]
            else: 
                tmpdata1[cnt][0][:] = im[ordertype[order][3]-1]
                tmpdata2[cnt][0][:] = im[ordertype[order][2]-1]
                tmpdata3[cnt][0][:] = im[ordertype[order][1]-1]
                tmpdata4[cnt][0][:] = im[ordertype[order][0]-1]
            
            self.label[cnt] =  order

            self.idxcounter = self.idxcounter + 1;
            if self.idxcounter == self.tuplenum:
                self.idxcounter = 0
                shuffle(self.randlist)


        top[0] = tmpdata1
        top[1] = tmpdata2
        top[2] = tmpdata3
        top[3] = tmpdata4
        top[4] = self.label

        return top
 

