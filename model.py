import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, len_feats, layers=[256], n_classes = 6, dropout=0.25,opts=None):
        super(Net, self).__init__()
        self.layers = layers
        if layers == [0]:
            model = [
                nn.Linear(len_feats, n_classes)
            ]
        else:
            #Input Layer
            model = [
                nn.Linear(len_feats,layers[0]),
                nn.BatchNorm1d(layers[0]),
                nn.ReLU()
                #nn.Dropout(dropout)
                
            ]
            #Hidden Layers
            for i, _ in enumerate(layers[:-1]):
                model += [
                    nn.Linear(layers[i],layers[i+1]),
                    nn.BatchNorm1d(layers[i+1]),
                    nn.ReLU()
                    #nn.Dropout(dropout)
                ]
            #Output Layer
            model += [
                nn.Linear(layers[-1], n_classes)
            ]
        self.num_params = 0
        self.model = nn.Sequential(*model)
        for param in self.model.parameters():
            self.num_params += param.numel()
        
    def forward(self, inp):

        res = self.model(inp)
        return  F.log_softmax(res, dim =-1)