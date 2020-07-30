
import argparse
import os
import json
import numpy as np
from pathlib import Path
from solver import Solver
from torch.backends import cudnn


def loadList(filename):
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()


def main(config):

    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    train_frame = loadList('./8/trainframe.npy') 
    train_file = loadList('./8/trainfile.npy') 
    test_frame = loadList('./8/testframe.npy')
    test_file = loadList('./8/testfile.npy')
    
    solver = Solver(config).train(train_file,train_frame,test_file,test_frame)
    cudnn.benchmark = True 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--imagesize', type=int, default=227)

    # training hyper-parameters
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--number_worker', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    
    # misc
    parser.add_argument('--model_path', type=str, default='./models_large')

    config = parser.parse_args()
    print(config)
    main(config)

