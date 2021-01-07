""" Model Architectures """

import torch
from torch.autograd import Variable

from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d



class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),

            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),   
            
            Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)  
        )

        self.linear_layers = Sequential(
            Linear(48, 2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train(epoch, model, optimizer, criterion, train_x, val_x, train_y, val_y):
    model.train()

    x_train, y_train = Variable(train_x), Variable(train_y)
    x_val, y_val = Variable(val_x), Variable(val_y)

    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    optimizer.zero_grad()

    """ introduce stretching here! """
    output_train = model(x_train)
    output_val = model(x_val)

    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    if epoch%2 == 0:
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
    
    return model