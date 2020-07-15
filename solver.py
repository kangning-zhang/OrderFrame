
import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
import random

from torch.autograd import Variable
from torch import optim
from model import OrderNetwork


class Solver(object):
    def __init__(self,config,train_data):
        self.train_data = train_data
        self.onet = None
        self.optimizer = None
        self.train_iter = config.train_iter
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.log_step = config.log_step
        self.model_path = config.model_path


    def reset_grad(self):
        self.optimizer.zero_grad()

    def train(self):

        self.onet = OrderNetwork()
        params = list(self.onet.parameters())
        self.optimizer = optim.Adam(params, self.lr, [self.beta1,self.beta2])

        if torch.cuda.is_available():
            self.onet.cuda()

        idx_pool = list(range(self.train_data[0].shape[0]))
        for step in range(self.train_iter + 1):
            idx = random.sample(idx_pool,self.batch_size)
            idx_pool = [i for j, i in enumerate(idx_pool) if j not in idx]
            if len(idx_pool) < self.batch_size:
                idx_pool = list(range(self.train_data[0].shape[0]))

            input1 = self.train_data[0][idx]
            input2 = self.train_data[1][idx]
            input3 = self.train_data[2][idx]
            input4 = self.train_data[3][idx]
            label = self.train_data[4][idx]

            if torch.cuda.is_available():
                input1, input2, input3, input4, label = Variable(input1).cuda(), Variable(input2).cuda(), Variable(input3).cuda(), Variable(input4).cuda, Variable(label).cuda()

            # ========Train Onet==========
            self.optimizer.zero_grad()

            output = self.onet(input1,input2,input3,input4)
            loss = nn.CrossEntropyLoss(output,label)
            loss.backward()
            self.optimizer.step()

            # =======Print the log info=========
            running_loss += loss.item()
            if (step + 1) % self.log_step == 0:
                print('Step [%d/%d], loss: %.3f'
                      %(step+1, self.train_iters, running_loss/self.log_step))
                running_loss = 0.00

            
            if (step+1) % 20 == 0:
                # save the model parameters for each epoch
                onet_path = os.path.join(self.model_path, 'iter-%d.pkl' %(step+1))
                torch.save(self.onet.state_dict(), onet_path)
