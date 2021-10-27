import torch
from torch import nn as nn
import torch.nn.functional as F
from itertools import combinations
from dod import DOD


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)



class AE_MNIST(nn.Module):
    def __init__(self,loss_type):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(784,500),nn.BatchNorm1d(500),nn.ReLU(),
            nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
           nn.Linear(500,2000),nn.BatchNorm1d(2000),nn.ReLU(),
           nn.Linear(2000,10)
        )
        self.dec = nn.Sequential(
            nn.Linear(10,2000),nn.BatchNorm1d(2000),nn.ReLU(),
            nn.Linear(2000,500),nn.BatchNorm1d(500),nn.ReLU(),
            nn.Linear(500,500),nn.BatchNorm1d(500),nn.ReLU(),
            nn.Linear(500,784),nn.Sigmoid()
        )

        self.enc1.apply(init_weights)
        self.enc2.apply(init_weights)
        self.dec.apply(init_weights)
        self.loss_type = loss_type

    def forward(self,x):
        n,c,h,w = x.shape
        a = self.enc1(x.reshape(n,-1))
        z = self.enc2(a)
        x_hat = self.dec(z)
        x_hat = x_hat.reshape(n,c,h,w)
        return x_hat,a,z

    def dod_loss(self,x,a,z):
        l = list(range(x.shape[0]))
        c = combinations(l,2)
        y = [i for i in c]
        y1=[]
        y2=[]
        for i,j in y:
            y1.append(i)
            y2.append(j)

        loss = DOD(
            x1=x[y1,:],
            x2=x[y2,:],
            a1=a[y1,:],
            a2=a[y2,:],
            z1=z[y1,:],
            z2=z[y2,:],
            dis =self.loss_type)()
        # b = x.shape[0]//2
        # loss = DOD(
        #     x1 = x[:b],x2 = x[b:] , a1 = a[:b] , a2 = a[b:] , z1 = z[:b] , z2 = z[b:] , dis = self.loss_type
        # )()
        return loss