import argparse
import os
from solver import Solver
from torch.backends import cudnn
from data_loader import train_loader


def main(config):
    train_data = train_loader(config).load()
    solver = Solver(config)
    cudnn.benchmark = True 
    
    # create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
        
    solver.train(train_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--imagesize1', type=int, default=288)
    parser.add_argument('--imagesize2', type=int, default=224)

    # training hyper-parameters
    parser.add_argument('--train_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # misc
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--log_step', type=int , default=10)

    config = parser.parse_args()
    print(config)
    main(config)
