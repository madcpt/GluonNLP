import random

import torch
from torch import nn
from torch.utils import data as Data

print(torch.__version__)


class Equation():
    def __init__(self, max_num=5, output_len=5, padding_len=5, device='cpu', *args):
        super(Equation, self).__init__(*args)
        self.idx2str = list(range(10))
        self.idx2str += ['+','<bos>','<pad>','<eos>']
        self.str2idx = {'+':10, '<bos>':10+1, '<pad>':10+2, '<eos>':10+3}
        self.data_set = []
        self.train_set_feature = []
        self.train_set_label = []
        self.test_set_feature = []
        self.test_set_label = []
        self.device = device
        for i in range(max_num):
            for j in range(max_num):
                num1 = [int(t) for t in str(i)]
                num2 = [int(t) for t in str(j)]
                equation = [*num1,self.str2idx['+'],*num2]
                while len(equation) < padding_len:
                    equation = equation + [self.str2idx['<pad>']]
                result = [int(t) for t in str(i+j)]
                result.append(self.str2idx['<eos>'])
                result = [self.str2idx['<bos>']] + result
                while len(result) < padding_len:
                    result = result + [self.str2idx['<pad>']]
                self.data_set.append((equation, result))
    
    def generate_train_test(self, test_sample_rate=0.2):
        self.train_set_feature = []
        self.train_set_label = []
        self.test_set_feature = []
        self.test_set_label = []
        for i in self.data_set:
            if random.random() < test_sample_rate:
                self.test_set_feature.append(i[0])
                self.test_set_label.append(i[1])
            else:
                self.train_set_feature.append(i[0])
                self.train_set_label.append(i[1])
        self.train_set_feature = torch.tensor(self.train_set_feature, device=self.device)
        self.train_set_label = torch.tensor(self.train_set_label, device=self.device)
        self.test_set_feature = torch.tensor(self.test_set_feature, device=self.device)
        self.test_set_label = torch.tensor(self.test_set_label, device=self.device)
        return 1

    def generate_data_iter(self, batch_size):
        '''
        Deprecated
        '''
        data_set_size = len(self.data_set)
        batch_num = data_set_size // batch_size
        for i in range(batch_num):
            X_raw = [pair[0] for pair in self.data_set[i*batch_size: i*batch_size+batch_size]]
            Y_raw = [pair[1] for pair in self.data_set[i*batch_size: i*batch_size+batch_size]]
            X = torch.tensor(X_raw, device=self.device)
            Y = torch.tensor(Y_raw, device=self.device)
            yield X, Y
    
    def generate_data_iter(self, batch_size, mode='train'):
        if mode == 'train':
            dataset = Data.TensorDataset(self.train_set_feature, self.train_set_label)
            data_iter = Data.DataLoader(dataset, batch_size, False, drop_last=True)
        if mode == 'test':
            dataset = Data.TensorDataset(self.test_set_feature, self.test_set_label)
            data_iter = Data.DataLoader(dataset, batch_size, False, drop_last=True)
        return data_iter


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    equ = Equation(99, device=device)
    equ.generate_train_test(0.1)
    print(len(equ.test_set_feature))
    print(len(equ.train_set_feature))

    # for x,y in equ.generate_data_iter(15):
    #     print(x)
    #     print(y)
