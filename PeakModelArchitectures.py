""" Peak Model Architectures """

import torch
from torch.autograd import Variable

from torch.nn import Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module, BatchNorm2d, Dropout, BatchNorm1d



class Net_p(Module):   
    def __init__(self):
        super(Net_p, self).__init__()

        self.cnn_layers = Sequential(
            Conv1d(1, 10, kernel_size=2, stride=1),
            BatchNorm1d(10),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=1),
            Dropout(0.1),

            Conv1d(10, 10, kernel_size=2, stride=1),
            BatchNorm1d(10),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=1),
            Dropout(0.1),
            
            Conv1d(10, 10, kernel_size=2, stride=1),
            BatchNorm1d(10),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=1),
            Dropout(0.1),
            
            Conv1d(10, 10, kernel_size=2, stride=1),
            BatchNorm1d(10),
            ReLU(),
            MaxPool1d(kernel_size=2, stride=1),
            Dropout(0.1),
        )

        self.linear_layers = Sequential(
            Linear(40, 2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train_p(epoch, model, optimizer, criterion, train_x, train_y):
    model.train()

    x_train, y_train = Variable(train_x), Variable(train_y)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()

    optimizer.zero_grad()

    """ introduce stretching here! """
    output_train = model(x_train)

    loss_train = criterion(output_train, y_train)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    if epoch%2 == 0:
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_train)
    
    return model