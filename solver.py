import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
import random
import json
import copy

from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from model_large import OrderNetwork
from data_loader_large import PulseDataset, ToTensor

def saveList(myList,filename):
    np.save(filename,myList) 


class Solver(object):
    def __init__(self,config):

        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.num_worker = config.number_worker
        self.lr = config.lr
        self.model_path = config.model_path

        self.onet = OrderNetwork()
        params = list(self.onet.parameters())
        self.optimizer = optim.SGD(params, lr = self.lr, momentum = 0.9)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=6,gamma=0.1)



    def train(self,train_file,train_frame,test_file,test_frame):

        traindata = PulseDataset(train_file, train_frame, transform=transforms.Compose([ToTensor()]))
        trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.batch_size,shuffle = True, num_workers = 2,drop_last=True)
        trainnum = len(train_file)

        testdata = PulseDataset(test_file, test_frame, transform=transforms.Compose([ToTensor()]))
        testloader = torch.utils.data.DataLoader(testdata, batch_size=self.batch_size,shuffle = True, num_workers = 2,drop_last=True)
        testnum = len(test_file)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.onet.to(device)

        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.onet.state_dict())

        train_acc = []
        train_loss = []
        test_acc = []
        test_loss = []

        for epoch in range(self.epoch):

            train_idx_pool = list(range(trainnum))
            test_idx_pool = list(range(testnum))

            train_running_loss = 0.0
            train_running_corrects = 0
            test_running_loss = 0.0
            test_running_corrects = 0
            

            for step, (frame1, frame2, frame3, frame4, label) in enumerate(trainloader):

                self.onet.train()
                input1 = frame1.to(device)
                input2 = frame2.to(device)
                input3 = frame3.to(device)
                input4 = frame4.to(device)
                label = label.to(device)


                # ========Train Onet==========
                self.optimizer.zero_grad()

                output = self.onet(input1,input2,input3,input4, self.batch_size)
                _, preds = torch.max(output, 1)
                loss = nn.CrossEntropyLoss()(output,label.long())
                loss.backward()
                self.optimizer.step()

                # =======Print the log info=========
                train_running_loss += loss.item() * label.size(0)
                train_running_corrects += torch.sum(preds == label)

                if (step%50==0):
                    print(step,'completed, loss = ', loss.item(),'acc = ',(torch.sum(preds == label).double() / 128))

            self.scheduler.step()
            remain = trainnum % 128
            epoch_loss = train_running_loss / (trainnum - remain)
            epoch_acc = train_running_corrects.double() / (trainnum - remain)
            train_acc.append(epoch_acc)
            train_loss.append(epoch_loss)

            print('Epoch [{}/{}]: train loss: {:.4f} train acc: {:.4f}'.format(epoch+1,self.epoch,epoch_loss,epoch_acc))


            for step, (frame1, frame2, frame3, frame4, label) in enumerate(testloader):

                with torch.set_grad_enabled(False):

                    self.onet.eval()
                    input1 = frame1.to(device)
                    input2 = frame2.to(device)
                    input3 = frame3.to(device)
                    input4 = frame4.to(device)
                    label = label.to(device)

                    output = self.onet(input1,input2,input3,input4, self.batch_size)
                    _, preds = torch.max(output, 1)
                    loss = nn.CrossEntropyLoss()(output,label.long())

                    test_running_loss += loss.item() * label.size(0)
                    test_running_corrects += torch.sum(preds == label)

                    if (step%50==0):
                        print(step,'completed')

            remain = testnum % 128
            epoch_loss = test_running_loss / (testnum - remain)
            epoch_acc = test_running_corrects.double() / (testnum - remain)
            test_acc.append(epoch_acc)
            test_loss.append(epoch_loss)

            print('Epoch [{}/{}]: val loss: {:.4f} val acc: {:.4f}'.format(epoch+1,self.epoch,epoch_loss,epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.onet.state_dict())
                
            # save the model parameters for each epoch and best epoch    

            onet_path = os.path.join(self.model_path, 'epoch-%d.pkl' %(epoch+1))
            torch.save(self.onet.state_dict(), onet_path)

        onet_path = os.path.join(self.model_path, 'best.pkl')
        torch.save(best_model_wts, onet_path)   

        torch.save(train_acc,os.path.join(self.model_path, 'train_acc.npy'))
        saveList(train_loss,os.path.join(self.model_path, 'train_loss.npy'))
        torch.save(test_acc,os.path.join(self.model_path, 'test_acc.npy'))
        saveList(test_loss,os.path.join(self.model_path, 'test_loss.npy'))

                
    