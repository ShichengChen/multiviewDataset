import numpy as np
import os
import cv2
import pickle
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
import torch.nn as nn


class VAE4Pose(nn.Module):
    def __init__(self, nn_dim = 256, latent_dim = 32, in_dim = 21*3, num_joints = 21,hand_side = 'right'):
        super(VAE4Pose, self).__init__()

        assert hand_side in ['left', 'right'], print('The type of hand_side input should be left or right')

        self.num_joints = num_joints

        self.in_size = in_dim
        self.num_neurons = nn_dim
        self.latent_size = latent_dim

        self.enc=nn.Sequential(
            nn.BatchNorm1d(self.in_size),
            nn.Linear(self.in_size, self.num_neurons),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.BatchNorm1d(self.num_neurons),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(self.num_neurons),
            nn.Linear(self.num_neurons, self.num_neurons),
        )

        self.dec = nn.Sequential(
            nn.Linear(self.latent_size, self.num_neurons),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.num_neurons, self.num_neurons),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.num_neurons, self.num_joints*3),
        )
        self.enc_mu = nn.Linear(self.num_neurons, self.latent_size)
        self.enc_logvar = nn.Linear(self.num_neurons, self.latent_size)
        self.hand_side = hand_side

    def encode(self, x,train=True):
        #print('x00',float(torch.sum(x!=x)))
        N=x.shape[0]
        x = self.enc(x.view(N, 63))
        #print('x01', float(torch.sum(x!=x)))
        if(train):
            m=torch.distributions.normal.Normal(self.enc_mu(x), F.softplus(self.enc_logvar(x)))
            q_z_sample = m.rsample()
            return q_z_sample,m
        else:
            return self.enc_mu(x)


    def decode(self, x):
        #print('x0',float(torch.sum(x!=x)))
        x=self.dec(x).view(-1, 21, 3)
        #print('x1', float(torch.sum(x!=x)))
        return x

    def forward(self, x,train=True):
        if (train):
            x,q_z = self.encode(x,train)
            y = self.decode(x)
            results = {'mean':q_z.mean, 'std':q_z.scale, 'out': y}
        else:
            x = self.encode(x,train)
            y = self.decode(x)
            results = y
        return results


if __name__ == '__main__':
    network = VAE4Pose(hand_side='right')
    input_d = torch.ones(4,21,3)
    out = network.forward(input_d)
    print(out['pose'].shape)
